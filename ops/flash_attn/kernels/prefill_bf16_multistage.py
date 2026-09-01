"""FlashAttention prefill — bf16, multi-stage, non-warpspec, varlen (Exercise 1).

The kernel implements FlashAttention v2 **varlen** prefill (hpc-ops compatible
API) with a multi-stage Hopper WGMMA pipeline. The batch dimension is implicit
in the ``seqlens`` / ``cu_seqlens`` arrays: inputs are flattened to
``(total_seq, H, D)`` (hpc-ops ``attention_prefill_bf16`` form) and every
sequence is always causal.

Class-based kernel following the Hopper FMHA pattern. Uses ``layout_acc_mn``
for 2D (M, N) C-fragment views and ``warp_reduction_max/sum`` for intra-warp
reduction.

**Key design point — N embedded in the MMA shape.** The QK instruction is
``(64, 64, 16)`` with atom layout ``(1, 1, 1)``: all 64 KV columns land in ONE
warpgroup (128 threads), and each M row's 64 N values are spread over just 4
threads of a warp (``reduction_target_n == (4,)``). ``warp_reduction`` then
covers the full softmax row.  Do NOT use ``(64,16,16)+(1,4,1)`` — that tiled
mma needs 512 threads; launched at 128 it computes only 1/4 of N and the
softmax row-sum collapses (the ~16x error seen in early iterations).

Algorithm (single-pass online softmax, per Hopper FMHA pattern)::

    for j_block in range(num_n_blocks):
        load K, V → smem
        S = Q @ K^T * scale                   # WGMMA (SS scope)
        m_new = max(m, rowmax(S))             # per-element + warp_reduction_max
        P = exp2(scale * S - scale * m_new)   # per-element
        O *= exp2(scale * (m - m_new))        # per-element rescale
        l = l * exp2(scale * (m - m_new)) + rowsum(P)   # a_sum, NO cross-thread
                                                         # reduction inside loop
        O += P @ V                            # WGMMA (RS scope: P in regs)
        m = m_new
    l = warp_reduction_sum(l)                 # cross-thread sum ONCE here
    O /= l                                    # normalize
    store O

The ``a_sum`` cross-thread sum must run only in the epilogue: calling
``warp_reduction_sum`` inside the KV loop re-sums the already-complete
running total on every block (after the first reduction, all 4 threads of a
row hold the full sum), over-counting it 4x per iteration.

Varlen semantics: the causal mask uses **batch-local** coordinates and gmem is
tiled at BLK_M granularity, so every batch's flattened start index
(``cu_seqlens[b]``) must be a multiple of BLK_M (64). With misaligned batch
boundaries tiles would straddle two batches and the mask would mismatch the
actual gmem rows (silent wrong results). The harness pads each batch's Q/K/O
segment to 64 rows so arbitrary real seqlens are supported.
"""

from __future__ import annotations

import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
from cuda.bindings.driver import CUstream
from cutlass import cute
from cutlass.cute.nvgpu.warpgroup import OperandMajorMode, OperandSource
from cutlass.utils.layout import LayoutEnum


BLK_M = 64
BLK_N = 64
D = 128
NUM_THREADS = 128


class FlashAttnPrefillBf16Multistage:
    """Varlen FlashAttention prefill kernel (bf16, single warpgroup, multi-stage).

    hpc-ops compatible: ``(total_seq, H_q/H_kv, D)`` + ``seqlens`` +
    ``cu_seqlens``, always causal, internal 1/sqrt(D) scale.  See module
    docstring for the 64-aligned batch boundary precondition.
    """

    def __init__(self):
        self.q_dtype = cutlass.BFloat16
        self.kv_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.o_dtype = cutlass.BFloat16
        # N (KV cols) inside the instruction shape → all 64 cols land in ONE
        # warpgroup (128 threads). (1,4,1)+(64,16,16) would need 512 threads;
        # launched at 128 it computes only 1/4 of N → broken softmax.
        self.qk_shape = (64, 64, 16)
        self.qk_atom_layout = (1, 1, 1)
        self.pv_atom_layout = (1, 1, 1)

    def layout_separate(self, thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())
        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])
        if cutlass.const_expr(cute.rank(lt) == 1):
            return cute.append(lt, ge)
        return cute.append(cute.append(cute.make_layout(()), lt), ge)

    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):
        separated = self.layout_separate(tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1])
        V_M = separated[0]
        V_N = separated[1]
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])
        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            return cute.append(V_M1, V_N1)
        return cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)

    @cute.jit
    def reduction_target_n(self, tiled_mma):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

    @staticmethod
    def convert_c_layout_to_a_layout(c, a):
        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c.stride[0],
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
            ),
        )

    @cute.jit
    def make_acc_into_op(self, acc, operand_layout_tv, Element):
        operand = cute.make_rmem_tensor_like(
            self.convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            Element,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        acc_vec = acc.load()
        operand_as_acc.store(acc_vec.to(Element))
        return operand

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mSeqlens: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        r2s_tiled_copy_o: cute.TiledCopy,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        scale_log2: cutlass.Float32,
        H_q: cutlass.Constexpr[int],
        H_kv: cutlass.Constexpr[int],
        max_seqlens: cutlass.Constexpr[int],
        Dd: cutlass.Constexpr[int],
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid_bh, bid_m, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP_flat = cute.make_tensor(storage.sP.data_ptr(), cute.make_layout((BLK_M, BLK_N)))
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        # Varlen: this CTA owns (batch = bid_bh // H_q, head = bid_bh % H_q).
        # gmem is (total_seq, H_q/H_kv, D) — NOT batch-folded: the head index
        # is h_q//gqa directly (a b*H_kv offset would read past H_kv heads).
        gqa = H_q // H_kv
        b = bid_bh // H_q
        h_q = bid_bh % H_q

        q_start = mCuSeqlens[(b,)]
        q_len = mSeqlens[(b,)]
        q_tile_start = bid_m * BLK_M
        # Absolute first row of this CTA's Q-tile in the flattened tensor.
        q_row0 = q_start + q_tile_start

        # Tiles beyond this batch's actual length must write nothing.  A staged
        # `if` cannot exit early, so the whole CTA body is wrapped below.
        if q_tile_start < q_len:
            # Varlen gmem layouts are (total_seq, H_q/H_kv, D): mode0 is the
            # (flattened) sequence dim, mode1 the head dim.  local_tile maps
            # tiler modes 1:1 onto input modes, so the tiler here is
            # (tile_rows, 1, Dd) — coord indices are *tile* indices,
            # i.e. absolute row // tile size.
            gQ = cute.local_tile(mQ, (BLK_M, 1, Dd), (q_row0 // BLK_M, h_q, 0))
            cute.autovec_copy(cute.group_modes(gQ, 0, 2), sQ)
            cute.arch.sync_threads()

            bf16 = self.q_dtype
            f32 = self.acc_dtype

            qk_thr = qk_tiled_mma.get_slice(tidx)
            tCsQ = qk_thr.partition_A(sQ)
            tCrQ = qk_tiled_mma.make_fragment_A(tCsQ)
            tCsS = qk_thr.partition_C(sP_flat)
            tCrS = qk_tiled_mma.make_fragment_C(tCsS)

            pv_thr = pv_tiled_mma.get_slice(tidx)
            tCsO = pv_thr.partition_C(sO)
            tCrO = pv_tiled_mma.make_fragment_C(tCsO)
            thr_r2s_o = r2s_tiled_copy_o.get_slice(tidx)

            acc_qk_mn = cute.make_tensor(tCrS.iterator, self.layout_acc_mn(qk_tiled_mma, tCrS.layout))
            acc_pv_mn = cute.make_tensor(tCrO.iterator, self.layout_acc_mn(pv_tiled_mma, tCrO.layout))
            qk_m = cute.size(acc_qk_mn, mode=[0])
            qk_n = cute.size(acc_qk_mn, mode=[1])
            pv_n = cute.size(acc_pv_mn, mode=[1])

            red_target = self.reduction_target_n(qk_tiled_mma)
            red_rank = cute.rank(red_target)

            s_max = cute.make_rmem_tensor((qk_m,), f32)
            a_sum = cute.make_rmem_tensor((qk_m,), f32)
            s_max_prev = cute.make_rmem_tensor((qk_m,), f32)
            s_max.fill(float("-inf"))
            a_sum.fill(0.0)
            tCrO.fill(0.0)

            # scale folded with log2(e) so exp2(x) computes exp(x*scale): one
            # less multiply per element vs. separate scale + exp.
            scale_val = (1.0 / (Dd**0.5)) * 1.4426950408889634
            num_k_blocks = cute.size(tCrQ, mode=[2])
            # Causal KV limit for this Q-tile: rows [0, q_tile_start + BLK_M)
            # within the batch, capped by the batch length.  (Always causal.)
            kv_limit = q_tile_start + BLK_M
            kv_limit = min(kv_limit, q_len)
            num_n_blocks = (kv_limit + BLK_N - 1) // BLK_N

            # Batch-aware causal mask per tile.  c_idx is a batch-local
            # identity tensor (max_s x max_s) routed through local_tile +
            # partition_C, so each fragment element carries its batch-local
            # (q_row, k_col) coordinate in [0, max_s).  All comparisons stay
            # batch-local (gK/gV are indexed from q_start, so the fragment's
            # columns are this batch's KV rows shifted by q_start).
            c_idx = cute.make_identity_tensor((max_seqlens, max_seqlens))

            for j_block in cutlass.range(num_n_blocks, unroll=1):
                # K rows live in gmem at [q_start + ...] — the batch's KV
                # starts at its own first query row (self-attn, seq_k==seq_q).
                gK = cute.local_tile(mK, (BLK_N, 1, Dd), ((q_start + j_block * BLK_N) // BLK_N, h_q // gqa, 0))
                cute.autovec_copy(cute.group_modes(gK, 0, 2), sK)
                # V is pre-transposed + per-batch padded to (B, H_kv, D, max_s)
                # by the harness so the PV B-operand (D, BLK_N) is K-major:
                # WGMMA computes P@V. Tile = (b, kv_h, 0:D, j_block*BLK_N+..).
                gV = cute.local_tile(mV, (1, 1, Dd, BLK_N), (b, h_q // gqa, 0, j_block))
                cute.autovec_copy(cute.group_modes(gV, 1, 3), sV)
                cute.arch.sync_threads()

                # Identity tile coord tracks j_block so k_col covers the
                # current KV block, not the first one.
                g_idx = cute.local_tile(c_idx, (BLK_M, BLK_N), (bid_m, j_block))
                tIdx = qk_thr.partition_C(g_idx)
                idx_mn = cute.make_tensor(tIdx.iterator, self.layout_acc_mn(qk_tiled_mma, tIdx.layout))

                tCsK = qk_thr.partition_B(sK)
                tCrK = qk_tiled_mma.make_fragment_B(tCsK)

                # QK WGMMA: S[m, n] = Q[m, :] @ K[n, :]^T. First k-block must
                # overwrite the accumulator; later blocks accumulate
                # (D/MMA-K loop).
                cute.nvgpu.warpgroup.fence()
                qk_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for k_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    cute.gemm(qk_tiled_mma, tCrS, tCrQ[None, None, k_idx], tCrK[None, None, k_idx], tCrS)
                    qk_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                # Batch-local causal mask: rows past q_len and upper-triangle
                # k_col > q_row set to -inf.
                for i in cutlass.range_constexpr(qk_m):
                    if idx_mn[i, 0][0] >= q_len:
                        for j in cutlass.range_constexpr(qk_n):
                            acc_qk_mn[i, j] = float("-inf")
                    else:
                        for j in cutlass.range_constexpr(qk_n):
                            k_col = idx_mn[i, j][1]
                            q_row = idx_mn[i, j][0]
                            if k_col >= q_len or k_col > q_row:
                                acc_qk_mn[i, j] = float("-inf")

                # Online softmax (FlashAttention v2): running max m, running
                # sum l.  m and l live in the QK C-space (qk_m values per
                # thread); the 4 threads sharing an M row combine via
                # warp_reduction_max so m is the *global* row max before P.
                for i in cutlass.range_constexpr(qk_m):
                    s_max_prev[i] = s_max[i]
                    for j in cutlass.range_constexpr(qk_n):
                        s_max[i] = cutlass.max(s_max[i], acc_qk_mn[i, j])
                    for r in cutlass.range_constexpr(red_rank):
                        s_max[i] = cute.arch.warp_reduction_max(s_max[i], threads_in_group=red_target.shape[r])

                    local_max = s_max[i]
                    if s_max[i] == float("-inf"):
                        local_max = 0.0

                    # P = exp2(scale*(S - m)); a_sum accumulates the row sums
                    # of P WITHOUT the cross-thread sum (runs once at the end).
                    for j in cutlass.range_constexpr(qk_n):
                        acc_qk_mn[i, j] = cute.math.exp2(
                            scale_val * acc_qk_mn[i, j] - scale_val * local_max,
                            fastmath=True,
                        )

                    # Rescale the running O and l by
                    # exp2(scale*(m_prev - m_new)); O and l are rescaled per
                    # element via the 2D acc_pv_mn view.
                    scale_pv = cute.math.exp2(
                        (s_max_prev[i] - local_max) * scale_val,
                        fastmath=True,
                    )
                    a_sum[i] = a_sum[i] * scale_pv
                    for j in cutlass.range_constexpr(pv_n):
                        acc_pv_mn[i, j] = acc_pv_mn[i, j] * scale_pv

                    a_sum[i] = a_sum[i] + acc_qk_mn[i, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0)

                # P (QK C-fragment) becomes the PV A-operand directly in
                # registers (RS scope) — no smem round-trip for P.
                # make_acc_into_op converts the C-layout to the A-operand
                # layout + casts fp32→bf16.
                acc_qk_fixed = self.make_acc_into_op(tCrS, pv_tiled_mma.tv_layout_A, bf16)
                tCsV = pv_thr.partition_B(sV)
                tCrV = pv_tiled_mma.make_fragment_B(tCsV)

                # PV WGMMA: O[m, d] += sum_n P[m, n] * V[n, d] (accumulate
                # across KV blocks).
                cute.nvgpu.warpgroup.fence()
                pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.gemm(pv_tiled_mma, tCrO, acc_qk_fixed, tCrV, tCrO)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                cute.arch.sync_threads()

            # a_sum is accumulated per-thread (local N cols) inside the loop;
            # the cross-thread sum happens once here (a row's N cols span 4
            # threads).  Doing it in-loop would re-sum the already-complete
            # total 4x per block.
            for i in cutlass.range_constexpr(qk_m):
                for r in cutlass.range_constexpr(red_rank):
                    a_sum[i] = cute.arch.warp_reduction_sum(a_sum[i], threads_in_group=red_target.shape[r])
                s = a_sum[i]
                inv = cute.arch.rcp_approx(s)
                if s == 0.0 or s != s:  # noqa: PLR0124
                    inv = 1.0
                for j in cutlass.range_constexpr(pv_n):
                    acc_pv_mn[i, j] = acc_pv_mn[i, j] * inv

            # Epilogue: O /= l, cast fp32→bf16, write the Q-tile back to gmem
            # via a register→gmem tiled copy (CopyUniversalOp route).
            gO = cute.local_tile(mO, (BLK_M, 1, Dd), (q_row0 // BLK_M, h_q, 0))
            acc_o_bf16 = cute.make_fragment_like(tCrO, self.o_dtype)
            acc_o_bf16.store(tCrO.load().to(self.o_dtype))
            tDrO = thr_r2s_o.retile(acc_o_bf16)
            tDgO = thr_r2s_o.partition_D(cute.group_modes(gO, 0, 2))
            cute.copy(r2s_tiled_copy_o, tDrO, tDgO)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mSeqlens: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        stream: CUstream,
        max_seqlens: cutlass.Constexpr[int],
        H_q: cutlass.Constexpr[int],
        H_kv: cutlass.Constexpr[int],
        Dd: cutlass.Constexpr[int],
    ):
        """hpc-ops compatible varlen prefill (always causal).

        MMA/SMEM setup identical across exercises; grid is
        (B * H_q, ceil(max_seqlens/BLK_M), 1) with per-batch guard.
        """
        bf16 = self.q_dtype
        f32 = self.acc_dtype
        # QK MMA: A=Q and B=K both from SMEM (SS scope), K-major. The N=64
        # dimension lives in the instruction shape — NOT in atom_layout — so
        # one warpgroup (128 threads) covers all KV columns (see docstring).
        op_qk = cute.nvgpu.warpgroup.MmaF16BF16Op(
            bf16, f32, self.qk_shape, OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K
        )
        qk_mma = cute.make_tiled_mma(op_qk, atom_layout_mnk=self.qk_atom_layout)
        # PV MMA: A=P from registers (RS scope) — avoids an smem round-trip
        # for the softmax output; B=V from SMEM, K-major (V pre-transposed).
        op_pv = cute.nvgpu.warpgroup.MmaF16BF16Op(
            bf16, f32, (64, Dd, 16), OperandSource.RMEM, OperandMajorMode.K, OperandMajorMode.K
        )
        pv_mma = cute.make_tiled_mma(op_pv, atom_layout_mnk=self.pv_atom_layout)

        # r2s tiled copy for the epilogue: C-fragment → gmem (CopyUniversalOp
        # handles the register layout; partition_D projects onto the gmem tile).
        universal = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(universal, bf16)
        r2s_o = cute.make_tiled_copy_C(copy_atom, pv_mma)

        # SMEM layouts: swizzled layout atoms created via hopper_helpers, then
        # tile_to_shape fixes the tile extent. q_atom (K-major, D contiguous)
        # is reused for Q/K/O; sV is (D, BLK_N) K-major for the PV B-operand.
        q_atom = sm90_utils.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, Dd), bf16
        )
        v_atom = sm90_utils.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, BLK_N), bf16
        )
        p_atom = sm90_utils.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, BLK_N), bf16
        )
        sQ_layout = cute.tile_to_shape(q_atom, (BLK_M, Dd), order=(0, 1))
        sK_layout = cute.tile_to_shape(q_atom, (BLK_N, Dd), order=(0, 1))
        sV_layout = cute.tile_to_shape(v_atom, (Dd, BLK_N), order=(0, 1))
        sP_layout = cute.tile_to_shape(p_atom, (BLK_M, BLK_N), order=(0, 1))
        sO_layout = cute.tile_to_shape(q_atom, (BLK_M, Dd), order=(0, 1))

        # Shared storage struct laid out by the cutlass SMEM allocator; every
        # buffer 1024-B aligned so TMA/WGMMA 128-bit requirements are met.
        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sV_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sP_layout)], 1024]
            sO: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sO_layout)], 1024]

        scale_log2 = cutlass.Float32((1.0 / (Dd**0.5)) * 1.4426950408889634)
        B = mSeqlens.shape[0]
        grid = (B * H_q, (max_seqlens + BLK_M - 1) // BLK_M, 1)

        self.kernel(
            qk_mma,
            pv_mma,
            mQ,
            mK,
            mV,
            mO,
            mSeqlens,
            mCuSeqlens,
            r2s_o,
            sQ_layout,
            sK_layout,
            sV_layout,
            sP_layout,
            sO_layout,
            scale_log2,
            H_q,
            H_kv,
            max_seqlens,
            Dd,
            SharedStorage,
        ).launch(grid=grid, block=(NUM_THREADS, 1, 1), stream=stream)


__all__ = ["BLK_M", "BLK_N", "NUM_THREADS", "D", "FlashAttnPrefillBf16Multistage"]
