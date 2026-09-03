"""FlashAttention prefill — bf16, multi-stage TMA pipelined, non-warpspec, varlen (Exercise 1).

The kernel implements FlashAttention v2 **varlen** prefill (hpc-ops compatible
API) with a **TMA multi-stage** Hopper pipeline — the same structure as
hpc-ops ``attention_prefill_bf16_multi_stage`` (NOT warp-specialized: one
128-thread warpgroup both issues TMA loads and runs the WGMMAs, overlapping
K/V loads with compute via a kStage ring of smem buffers + mbarriers).

* **Loads**: all three of Q/K/V are TMA bulk-tensor loads issued by warp 0
  (``elect_one`` leader). Q is loaded once behind a single mbarrier; K and V
  stream through ``kStage`` ring buffers, each with its own full barrier, and
  the consumer waits ``bar_k`` before the QK MMA and ``bar_v`` before the PV
  MMA (hpc-ops ordering).

* **MMAs**: QK WGMMA ``(64,64,16)`` SS-scope, PV WGMMA ``(64,128,16)``
  RS-scope (P in registers, V from smem K-major) — identical to the v1
  kernel; only the loading path changed.

* **Causal**: every batch is always causal; identities and masks are
  batch-local (see module preconditions below). TMA zero-fills out-of-bounds
  tiles, so the padded tail (pack_varlen 64-aligns every batch's flattened
  Q/K/O segment) is safe to read.

Varlen semantics: the causal mask uses **batch-local** coordinates and gmem is
tiled at BLK_M granularity, so every batch's flattened start index
(``cu_seqlens[b]``) must be a multiple of BLK_M (64). The harness pads each
batch's Q/K/O segment to 64 rows so arbitrary real seqlens are supported.

Pipeline sketch (mirrors hpc-ops multi_stage_dim128)::

    prologue: leader TMA-loads Q (bar_q), then K/V tiles 0..kStage-2
              (bar_k[0..kStage-2], bar_v[0..kStage-2])
    for itile in range(num_n_blocks):
        leader: if itile_write < num_n_blocks: TMA-load K+V tile itile_write
                into stage itile_write % kStage (bar_k/bar_v)
        wait bar_k[ismem_read]
        S = Q @ K^T * scale                    # QK WGMMA (SS scope)
        m_new = max(m, rowmax(S))
        P = exp2(scale * S - scale * m_new)
        O *= exp2(scale * (m - m_new))
        l = l * exp2(scale * (m - m_new)) + rowsum(P)
        wait bar_v[ismem_read]
        O += P @ V                            # PV WGMMA (RS scope)
        m = m_new
        ismem_read = (ismem_read + 1) % kStage; flip phase at wrap
    l = cross-thread sum(l); O /= l; store O
"""

from __future__ import annotations

import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
from cuda.bindings.driver import CUstream
from cutlass import cute, pipeline
from cutlass.cute.nvgpu.warpgroup import OperandMajorMode, OperandSource
from cutlass.utils.layout import LayoutEnum


BLK_M = 64
BLK_N = 64
D = 128
NUM_THREADS = 128
NUM_STAGES = 2  # K/V ring depth. v3 A/B said "neutral" — but that pre-dated
# v5 (async Q) removing the per-CTA fixed cost that masked the overlap. The v6
# (stages x split) matrix shows stages=2 >= stages=1 on every multi-stage shape
# (e.g. 512^2: 1.04x vs 1.02x vs hpc; GQA512: 1.33x vs 0.99x). See PERFLOG Step 6.


class FlashAttnPrefillBf16Multistage:
    """TMA multi-stage varlen FlashAttention prefill (bf16, single warpgroup).

    hpc-ops compatible: ``(total_seq, H_q/H_kv, D)`` + ``seqlens`` +
    ``cu_seqlens``, always causal, internal 1/sqrt(D) scale.  See module
    docstring for the 64-aligned batch boundary precondition.
    """

    def __init__(self, num_stages: int = NUM_STAGES, split_k: int = 1):
        self.num_stages = num_stages
        # split-KV: the KV range of each Q-tile is cut into `split_k` disjoint
        # spans handled by independent CTAs (grid z), merged by an LSE-combine
        # kernel.  1 = classic single-CTA-per-tile fused path (no partials).
        self.split_k = split_k
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
        tma_atom_q: cute.CopyAtom,
        mQ_tma: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_tma: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_tma: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_tma: cute.Tensor,
        mSeqlens: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        mPO: cute.Tensor,
        mPm: cute.Tensor,
        mPl: cute.Tensor,
        r2s_tiled_copy_o: cute.TiledCopy,
        r2s_tiled_copy_po: cute.TiledCopy,
        sQ_layout_staged: cute.ComposedLayout,
        sK_layout_staged: cute.ComposedLayout,
        sV_layout_staged: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        scale_log2: cutlass.Float32,
        H_q: cutlass.Constexpr[int],
        H_kv: cutlass.Constexpr[int],
        max_seqlens: cutlass.Constexpr[int],
        Dd: cutlass.Constexpr[int],
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # grid = (q_tile, batch*head): the FAST dim is the Q-tile so a wave of
        # concurrent CTAs reads nested causal KV prefixes of the SAME head
        # (L2 reuse), matching hpc-ops' (tile_m, head, batch) launch order.
        bid_m, bid_bh, s_idx = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        f32 = self.acc_dtype

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)

        sQ_full = storage.sQ.get_tensor(sQ_layout_staged.outer, swizzle=sQ_layout_staged.inner)
        sK_full = storage.sK.get_tensor(sK_layout_staged.outer, swizzle=sK_layout_staged.inner)
        sV_full = storage.sV.get_tensor(sV_layout_staged.outer, swizzle=sV_layout_staged.inner)
        # sP_flat is a compile-time partition_C TEMPLATE only (tCrS lives in
        # registers), so it borrows sV's base pointer and owns no smem (v6b).
        sP_flat = cute.make_tensor(storage.sV.data_ptr(), cute.make_layout((BLK_M, BLK_N)))
        # v8: sO is a REAL buffer again (r2s landing pad for the TMA store)
        # but aliases the sQ region — the last sQ read is the final QK WGMMA
        # (wait_group(0) before any epilogue runs), and a sync_threads below
        # orders the overwrite. Same (BLK_M, Dd) bf16 swizzled layout as Q.
        # PLAIN row-major on purpose: r2s (partition_D) and the S2G TMA atom
        # must see byte-identical layouts; the swizzled sO_layout caused a
        # d>=64 column-swap between the two interpretations (v8 debug).
        sO = cute.make_tensor(storage.sQ.data_ptr(), cute.make_layout((BLK_M, Dd), stride=(Dd, 1)))

        # Varlen: this CTA owns (batch = bid_bh // H_q, head = bid_bh % H_q).
        # gmem is (total_seq, H_q/H_kv, D) — NOT batch-folded: the head index
        # is h_q//gqa directly (a b*H_kv offset would read past H_kv heads).
        gqa = H_q // H_kv
        b = bid_bh // H_q
        h_q = bid_bh % H_q
        h_kv = h_q // gqa

        q_start = mCuSeqlens[(b,)]
        q_len = mSeqlens[(b,)]
        q_tile_start = bid_m * BLK_M
        # Absolute first row of this CTA's Q-tile in the flattened tensor.
        q_row0 = q_start + q_tile_start

        # Tiles beyond this batch's actual length must write nothing.  A staged
        # `if` cannot exit early, so the whole CTA body is wrapped below.
        if q_tile_start < q_len:
            # ----- TMA pipeline (barriers live in SharedStorage) -----
            kv_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.bar_kv_array.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_THREADS // 32),
                tx_count=BLK_N * Dd * self.q_dtype.width // 8
                + BLK_N * Dd * self.q_dtype.width // 8,  # K tile + V tile bytes
                defer_sync=True,
                cta_layout_vmnk=cta_layout_vmnk,
            )
            # Single-stage pipeline for the once-per-CTA Q tile (v5 async Q).
            q_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.q_bar_array.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_THREADS // 32),
                tx_count=BLK_M * Dd * self.q_dtype.width // 8,
                defer_sync=True,
                cta_layout_vmnk=cta_layout_vmnk,
            )
            # Fence the (deferred) mbarrier init once, then sync all threads.
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            kv_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            kv_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            q_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            q_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            # ----- TMA partitions (3D gmem views, hpc-ops pattern) -----
            # K view is (S, D, H_kv): flat_divide by the 2D box tiler
            # (BLK_N, Dd) splits the leading (S, D) modes into (tile, rest);
            # group_modes(0, 3) folds the tile dims into one mode, and the
            # trailing (S-tiles, ...) modes stay indexable per tile.
            tSgK = cute.flat_divide(mK_tma, (BLK_N, Dd))
            tKsK, tKgK_tmp = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK_full, 0, 2),
                cute.group_modes(tSgK, 0, 2),
            )
            tKgK = tKgK_tmp[(None, None, 0, h_kv)]
            tSgV = cute.flat_divide(mV_tma, (Dd, BLK_N))
            tVsV, tVgV_tmp = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                0,
                cute.make_layout(1),
                cute.group_modes(sV_full, 0, 2),
                cute.group_modes(tSgV, 0, 2),
            )
            tVgV = tVgV_tmp[(None, None, None, b * H_kv + h_kv)]
            tSgQ = cute.flat_divide(mQ_tma, (BLK_M, Dd))
            tQsQ, tQgQ_tmp = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ_full, 0, 2),
                cute.group_modes(tSgQ, 0, 2),
            )
            tQgQ = tQgQ_tmp[(None, None, 0, h_q)]
            # v8: output tile for the epilogue TMA store (S2G atom over the
            # (S, D, H_q) view; gmem tile index q_row0//BLK_M at store time).
            tSgO = cute.flat_divide(mO_tma, (BLK_M, Dd))
            tOsO, tOgO_tmp = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                0,
                cute.make_layout(1),
                cute.group_modes(sO, 0, 2),
                cute.group_modes(tSgO, 0, 2),
            )
            tOgO = tOgO_tmp[(None, None, 0, h_q)]

            # ----- MMA fragments & softmax state (unchanged from v1) -----
            qk_thr = qk_tiled_mma.get_slice(tidx)
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

            # Causal KV extent for this Q-tile: rows [0, q_tile_start + BLK_M)
            # within the batch, capped by the batch length.
            kv_limit = q_tile_start + BLK_M
            kv_limit = min(kv_limit, q_len)
            num_n_blocks = (kv_limit + BLK_N - 1) // BLK_N
            num_tile_kv = (q_len + BLK_N - 1) // BLK_N  # total padded KV tiles (mask guard)
            # split-KV: this CTA (grid z = s_idx) owns tile span [kv_lo, kv_hi).
            # The split is disjoint (zero redundant compute); an LSE combine
            # kernel merges the partials afterwards.  split_k == 1 gives the
            # fused whole-range path.
            kv_lo = (s_idx * num_n_blocks) // self.split_k
            kv_hi = ((s_idx + 1) * num_n_blocks) // self.split_k
            # KV tile rows live in gmem at [q_start + itile*BLK_N ..) — gmem is
            # zero-padded past q_len so TMA reads of the tail tile are safe.
            kv_base = q_start // BLK_N  # absolute gmem tile index of this batch's K

            # Batch-aware causal mask per tile.  c_idx is a batch-local
            # identity tensor (max_s x max_s) routed through local_tile +
            # partition_C, so each fragment element carries its batch-local
            # (q_row, k_col) coordinate in [0, max_s).
            c_idx = cute.make_identity_tensor((max_seqlens, max_seqlens))

            # KV tiles below this one are entirely under the causal diagonal
            # (their last column (itile+1)*64-1 < q_tile_start = bid_m*64),
            # so the mask loop only runs from the diagonal tile onward.
            num_tile_full = bid_m

            # =================================================================
            # PROLOGUE — Q via TMA (async; waited before the mainloop), then
            # K/V tiles 0..kStage-2 prefetched by the leader warp.
            # =================================================================
            if warp_idx == 0:
                q_pipeline.producer_acquire(q_prod_state)
                cute.copy(
                    tma_atom_q,
                    tQgQ[None, q_row0 // BLK_M],
                    tQsQ[None, 0],
                    tma_bar_ptr=q_pipeline.producer_get_barrier(q_prod_state),
                )
                q_pipeline.producer_commit(q_prod_state)

            # K/V prologue: prefetch this CTA's first kStage-1 tiles (if any).
            if warp_idx == 0:
                for istage in cutlass.range(self.num_stages - 1, unroll=1):
                    if kv_lo + istage < kv_hi:
                        kv_pipeline.producer_acquire(kv_prod_state)
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_base + kv_lo + istage],
                            tKsK[None, kv_prod_state.index],
                            tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_prod_state),
                        )
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, 0, kv_lo + istage],
                            tVsV[None, kv_prod_state.index],
                            tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_prod_state),
                        )
                        kv_pipeline.producer_commit(kv_prod_state)
                    kv_prod_state.advance()

            # Q fragment (smem) for the QK SS-scope MMA — stage-free, built once.
            tCsQ = qk_thr.partition_A(sQ_full[None, None, 0])
            tCrQ = qk_tiled_mma.make_fragment_A(tCsQ)
            num_k_blocks = cute.size(tCrQ, mode=[2])

            # Wait for the Q TMA exactly once — sQ is never overwritten, so no
            # consumer_release is needed (all warps have arrived on empty after
            # this wait returns; later reads are pure consumers).
            q_pipeline.consumer_wait(q_cons_state)

            # =================================================================
            # MAINLOOP — K/V ring: producer prefetches, consumer computes.
            # =================================================================
            ismem_read = 0
            for jt in cutlass.range(kv_hi - kv_lo, unroll=1):
                itile = kv_lo + jt
                # Producer: prefetch tile itile + (kStage-1) into its ring slot
                # (warp 0 only; the TMA engine does the DMA in the background).
                if warp_idx == 0:
                    itile_write = itile + (self.num_stages - 1)
                    if itile_write < kv_hi:
                        kv_pipeline.producer_acquire(kv_prod_state)
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_base + itile_write],
                            tKsK[(None, kv_prod_state.index)],
                            tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_prod_state),
                        )
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, 0, itile_write],
                            tVsV[(None, kv_prod_state.index)],
                            tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_prod_state),
                        )
                        kv_pipeline.producer_commit(kv_prod_state)
                    kv_prod_state.advance()

                # Consumer: wait for the K tile it's about to read.
                kv_pipeline.consumer_wait(kv_cons_state)
                ismem_now = kv_cons_state.index

                # K fragment from the current ring slot (SS-scope QK MMA).
                sK = sK_full[None, None, ismem_now]
                tCsK = qk_thr.partition_B(sK)
                tCrK = qk_tiled_mma.make_fragment_B(tCsK)

                # Identity tile coord tracks itile so k_col covers the current
                # KV block, not the first one.
                g_idx = cute.local_tile(c_idx, (BLK_M, BLK_N), (bid_m, itile))
                tIdx = qk_thr.partition_C(g_idx)
                idx_mn = cute.make_tensor(tIdx.iterator, self.layout_acc_mn(qk_tiled_mma, tIdx.layout))

                # QK WGMMA: S[m, n] = Q[m, :] @ K[n, :]^T. First k-block must
                # overwrite the accumulator; later blocks accumulate.
                cute.nvgpu.warpgroup.fence()
                qk_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for k_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    cute.gemm(qk_tiled_mma, tCrS, tCrQ[None, None, k_idx], tCrK[None, None, k_idx], tCrS)
                    qk_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                # Batch-local causal mask: rows past q_len and upper-triangle
                # k_col > q_row set to -inf.  Skipped entirely for fully
                # unmasked tiles below the diagonal (see num_tile_full).
                if itile >= num_tile_full:
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
                acc_qk_fixed = self.make_acc_into_op(tCrS, pv_tiled_mma.tv_layout_A, self.q_dtype)

                # Consumer: V for this tile is ready only now — wait bar_v
                # (hpc-ops orders the V wait after the QK/softmax work).
                sV = sV_full[None, None, ismem_now]
                tCsV = pv_thr.partition_B(sV)
                tCrV = pv_tiled_mma.make_fragment_B(tCsV)

                # PV WGMMA: O[m, d] += sum_n P[m, n] * V[n, d] (accumulate
                # across KV blocks).
                cute.nvgpu.warpgroup.fence()
                pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.gemm(pv_tiled_mma, tCrO, acc_qk_fixed, tCrV, tCrO)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                kv_pipeline.consumer_release(kv_cons_state)
                kv_cons_state.advance()

            # a_sum is accumulated per-thread (local N cols) inside the loop;
            # the cross-thread sum happens once here (a row's N cols span 4
            # threads).  Doing it in-loop would re-sum the already-complete
            # total 4x per block.
            for i in cutlass.range_constexpr(qk_m):
                for r in cutlass.range_constexpr(red_rank):
                    a_sum[i] = cute.arch.warp_reduction_sum(a_sum[i], threads_in_group=red_target.shape[r])

            if cutlass.const_expr(self.split_k == 1):
                # Fused epilogue (v8): O /= l, cast fp32→bf16, r2s into the
                # sQ-aliased smem, then one bulk TMA store smem→gmem (the
                # lesson-11 S2G pattern: fence_proxy so the async proxy sees
                # the writes, sync, warp0 elect_one issues the store and
                # waits the read-back before CTA exit frees the smem).
                for i in cutlass.range_constexpr(qk_m):
                    s = a_sum[i]
                    inv = cute.arch.rcp_approx(s)
                    if s == 0.0 or s != s:  # noqa: PLR0124
                        inv = 1.0
                    for j in cutlass.range_constexpr(pv_n):
                        acc_pv_mn[i, j] = acc_pv_mn[i, j] * inv
                acc_o_bf16 = cute.make_fragment_like(tCrO, self.o_dtype)
                acc_o_bf16.store(tCrO.load().to(self.o_dtype))
                tDrO = thr_r2s_o.retile(acc_o_bf16)
                tDsO = thr_r2s_o.partition_D(sO)
                cute.copy(r2s_tiled_copy_o, tDrO, tDsO)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.copy(tma_atom_o, tOsO[None], tOgO[None, q_row0 // BLK_M])
                        # v10: no commit/wait_group here.  The fused CTA exits
                        # right after and never rewrites the sO pad, and PTX
                        # guarantees outstanding bulk async ops complete
                        # before kernel exit — waiting only delayed retire
                        # (~3-4% on dense long-seq grids).
            else:
                # split-KV partial epilogue: write the UN-normalized fp32 O
                # tile to PO, plus per-row running max (Pm) and running sum
                # (Pl).  An LSE-combine kernel merges the split_k partials.
                # Empty-span CTAs (kv_lo==kv_hi) naturally write O=0, m=-inf,
                # l=0, which the combine weights to zero.
                thr_r2s_po = r2s_tiled_copy_po.get_slice(tidx)
                mPO_s = mPO[(None, None, s_idx, None)]
                gPO = cute.local_tile(mPO_s, (BLK_M, 1, Dd), (q_row0 // BLK_M, h_q, 0))
                tDrPO = thr_r2s_po.retile(tCrO)
                tDgPO = thr_r2s_po.partition_D(cute.group_modes(gPO, 0, 2))
                cute.copy(r2s_tiled_copy_po, tDrPO, tDgPO)
                g_idx0 = cute.local_tile(c_idx, (BLK_M, BLK_N), (bid_m, 0))
                tIdx0 = qk_thr.partition_C(g_idx0)
                idx0 = cute.make_tensor(tIdx0.iterator, self.layout_acc_mn(qk_tiled_mma, tIdx0.layout))
                for i in cutlass.range_constexpr(qk_m):
                    row = q_start + idx0[i, 0][0]
                    mPm[(row, h_q, s_idx)] = s_max[i]
                    mPl[(row, h_q, s_idx)] = a_sum[i]

    @cute.kernel
    def combine_kernel(
        self,
        mPO: cute.Tensor,  # (T_pad, H_q, S, D) fp32 partial O (un-normalized)
        mPm: cute.Tensor,  # (T_pad, H_q, S) fp32 partial row max
        mPl: cute.Tensor,  # (T_pad, H_q, S) fp32 partial row sum
        mO: cute.Tensor,  # (T_pad, H_q, D) bf16 final output
        scale_log2: cutlass.Float32,
        SplitK: cutlass.Constexpr[int],
        H_q: cutlass.Constexpr[int],
        Dd: cutlass.Constexpr[int],
    ):
        """LSE merge across split-KV partials. One CTA per (row, head);
        thread d owns output column d.  O = sum_s w_s*PO_s / sum_s w_s*Pl_s
        with w_s = exp2(scale_log2*(m_s - M)), M = max_s m_s (w_s=0 when
        m_s=-inf, all-inf rows emit 0)."""
        # v7: thread (tx, ty) merges row = bx*ROWS + ty across cols
        # [tx*VEC, (tx+1)*VEC) so PO loads / O stores are VEC-wide vectors.
        # T_pad is 64-aligned, so ROWS divides the row grid exactly.
        VEC: cutlass.Constexpr[int] = 4
        ROWS: cutlass.Constexpr[int] = 4
        tx, ty, _ = cute.arch.thread_idx()
        bx, h, _ = cute.arch.block_idx()
        row = bx * ROWS + ty
        d0 = tx * VEC

        m_max = mPm[(row, h, 0)]
        for s in cutlass.range_constexpr(1, SplitK):
            m_max = cutlass.max(m_max, mPm[(row, h, s)])

        acc = cute.make_rmem_tensor((VEC,), cutlass.Float32)
        for j in cutlass.range_constexpr(VEC):
            acc[j] = cutlass.Float32(0.0)
        if m_max != float("-inf"):
            l_sum = cutlass.Float32(0.0)
            frag = cute.make_rmem_tensor((VEC,), cutlass.Float32)
            for s in cutlass.range_constexpr(SplitK):
                m_s = mPm[(row, h, s)]
                w = cute.math.exp2(scale_log2 * (m_s - m_max), fastmath=True)
                if m_s == float("-inf"):
                    w = cutlass.Float32(0.0)
                l_sum = l_sum + w * mPl[(row, h, s)]
                tile = cute.local_tile(mPO, (1, 1, 1, VEC), (row, h, s, tx))
                cute.autovec_copy(tile, frag)
                for j in cutlass.range_constexpr(VEC):
                    acc[j] = acc[j] + w * frag[j]
            inv = cute.arch.rcp_approx(l_sum)
            if l_sum == 0.0 or l_sum != l_sum:  # noqa: PLR0124
                inv = 1.0
            for j in cutlass.range_constexpr(VEC):
                acc[j] = acc[j] * inv
        out = cute.make_rmem_tensor((VEC,), self.o_dtype)
        out.store(acc.load().to(self.o_dtype))
        o_tile = cute.local_tile(mO, (1, 1, VEC), (row, h, tx))
        cute.autovec_copy(out, o_tile)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mSeqlens: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        mPO: cute.Tensor,
        mPm: cute.Tensor,
        mPl: cute.Tensor,
        stream: CUstream,
        max_seqlens: cutlass.Constexpr[int],
        H_q: cutlass.Constexpr[int],
        H_kv: cutlass.Constexpr[int],
        Dd: cutlass.Constexpr[int],
    ):
        """hpc-ops compatible varlen prefill (always causal), TMA multi-stage.

        MMA/SMEM setup identical across exercises; grid is
        (ceil(max_seqlens/BLK_M), B * H_q, split_k) with per-batch guard;
        split_k > 1 additionally launches the LSE combine kernel.
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
        # _o writes the bf16 output (fused path); _po writes the fp32 partial
        # (split-KV path) with the same C-fragment tiling.
        universal = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(universal, bf16)
        r2s_o = cute.make_tiled_copy_C(copy_atom, pv_mma)
        copy_atom_po = cute.make_copy_atom(universal, f32)
        r2s_po = cute.make_tiled_copy_C(copy_atom_po, pv_mma)

        # SMEM layouts: swizzled layout atoms created via hopper_helpers, then
        # tile_to_shape fixes the tile extent. q_atom (K-major, D contiguous)
        # is reused for Q/K/O; sV is (D, BLK_N) K-major for the PV B-operand.
        # K/V get a leading STAGE axis (ring buffers) for the pipeline.
        q_atom = sm90_utils.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, Dd), bf16
        )
        v_atom = sm90_utils.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, BLK_N), bf16
        )
        p_atom = sm90_utils.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, BLK_N), bf16
        )
        sQ_layout_staged = cute.tile_to_shape(q_atom, (BLK_M, Dd, 1), order=(0, 1, 2))
        sK_layout_staged = cute.tile_to_shape(q_atom, (BLK_N, Dd, self.num_stages), order=(0, 1, 2))
        sV_layout_staged = cute.tile_to_shape(v_atom, (Dd, BLK_N, self.num_stages), order=(0, 1, 2))
        sP_layout = cute.tile_to_shape(p_atom, (BLK_M, BLK_N), order=(0, 1))
        sO_layout = cute.tile_to_shape(q_atom, (BLK_M, Dd), order=(0, 1))

        # TMA atoms over 3D gmem views, mirroring hpc-ops multi_stage_dim128:
        #   Q/K: (total_seq*H) rows x D cols, H head dim → (S, D, H) view
        #   V:   K-major (D, S) tiles, H*B head dim → (D, S, BH) view
        # The tma tile box is 2D; the trailing H (or BH) mode stays outside
        # the box and is indexed per-CTA (one head per CTA, V folded BH).
        total_seq = mQ.shape[0]
        # Q view (S, D, H_q) — same construct as K, box (BLK_M, Dd), v5 async.
        q_view = cute.make_tensor(
            mQ.iterator,
            cute.make_layout((total_seq, Dd, H_q), stride=(H_q * Dd, 1, Dd)),
        )
        tma_atom_q, tma_tensor_q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            q_view,
            cute.slice_(sQ_layout_staged, (None, None, 0)),
            (BLK_M, Dd),
        )
        # v8: output view (S, D, H_q) + S2G atom; same construct as q_view but
        # CopyBulkTensorTileS2GOp, plain (non-staged) sO layout.
        o_view = cute.make_tensor(
            mO.iterator,
            cute.make_layout((total_seq, Dd, H_q), stride=(H_q * Dd, 1, Dd)),
        )
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            o_view,
            cute.make_layout((BLK_M, Dd), stride=(Dd, 1)),
            (BLK_M, Dd),
        )
        # K view (S, D, H_kv), strides (H_kv*D, 1, D): matches hpc-ops' K =
        # (max_seq, D, H_kv) construct.  The 2D tma box (BLK_N, Dd) divides the
        # leading (S, D) modes; the trailing H_kv mode is indexed per-CTA.
        k_view = cute.make_tensor(
            mK.iterator,
            cute.make_layout((total_seq, Dd, H_kv), stride=(H_kv * Dd, 1, Dd)),
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            k_view,
            cute.slice_(sK_layout_staged, (None, None, 0)),
            (BLK_N, Dd),
        )
        # V: gmem (B, H_kv, D, S) from pack_varlen → view (Dd, S, BH) with
        # strides (S, 1, Dd*S): the box (Dd, BLK_N) divides leading (D, S),
        # the trailing BH mode is indexed per-CTA.
        B_bh = mSeqlens.shape[0] * H_kv
        v_view = cute.make_tensor(
            mV.iterator,
            cute.make_layout((Dd, max_seqlens, B_bh), stride=(max_seqlens, 1, Dd * max_seqlens)),
        )
        tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            v_view,
            cute.slice_(sV_layout_staged, (None, None, 0)),
            (Dd, BLK_N),
        )

        # Shared storage: K/V ring buffers (staged cosize) plus Q and K/V
        # mbarrier arrays (PipelineTmaAsync needs 2 Int64 per stage). The P
        # C-fragment template owns no smem (v6b: compile-time layout algebra,
        # borrows sV's pointer); v7 deleted sO's MemRange the same way, and v8
        # reintroduces sO data as an r2s landing pad that ALIASES the sQ region
        # (Q is fully consumed before the epilogue writes it) — so the smem
        # budget stays at sQ + K/V ring with no extra buffers.
        @cute.struct
        class SharedStorage:
            bar_kv_array: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages]
            q_bar_array: cute.struct.MemRange[cutlass.Int64, 2]
            sQ: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sQ_layout_staged)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sK_layout_staged)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sV_layout_staged)], 1024]

        # Non-clustered CTA: vmnk = (1, 1, 1, 1).
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))

        scale_log2 = cutlass.Float32((1.0 / (Dd**0.5)) * 1.4426950408889634)
        B = mSeqlens.shape[0]
        grid = ((max_seqlens + BLK_M - 1) // BLK_M, B * H_q, self.split_k)

        self.kernel(
            qk_mma,
            pv_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            mSeqlens,
            mCuSeqlens,
            mPO,
            mPm,
            mPl,
            r2s_o,
            r2s_po,
            sQ_layout_staged,
            sK_layout_staged,
            sV_layout_staged,
            sP_layout,
            sO_layout,
            cta_layout_vmnk,
            scale_log2,
            H_q,
            H_kv,
            max_seqlens,
            Dd,
            SharedStorage,
        ).launch(grid=grid, block=(NUM_THREADS, 1, 1), stream=stream)

        # One CTA per padded (row, head): merge the split-KV partials with an
        # LSE combine (exp2-rescaled weighted sum of PO / weights of Pl).
        if cutlass.const_expr(self.split_k > 1):
            self.combine_kernel(
                mPO,
                mPm,
                mPl,
                mO,
                scale_log2,
                self.split_k,
                H_q,
                Dd,
            ).launch(
                grid=((mO.shape[0] + 3) // 4, H_q, 1),
                block=(Dd // 4, 4, 1),  # tx: column-vec group, ty: row group
                stream=stream,
            )


__all__ = ["BLK_M", "BLK_N", "NUM_THREADS", "D", "FlashAttnPrefillBf16Multistage"]
