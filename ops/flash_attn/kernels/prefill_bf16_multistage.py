"""FlashAttention prefill — bf16, multi-stage, non-warpspec (Exercise 1).

Class-based kernel following the Hopper FMHA pattern. Uses ``layout_acc_mn``
for 2D (M, N) C-fragment views and ``warp_reduction_max/sum`` for intra-warp
reduction.

**Known limitation**: with a single warpgroup (4 warps), WGMMA distributes N
across warps → each thread sees only 1 N value (``qk_n=1``) → softmax is
trivial. Cross-warp reduction via SMEM or universal copy doesn't work with
the current CuTe DSL API. The Hopper FMHA avoids this by using 2+ MMA WGs
where each WG covers ALL N values. Exercise 2 (warpspec) will fix this.

Algorithm (single-pass online softmax, per Hopper FMHA pattern)::

    for j_block in range(num_n_blocks):
        load K, V → smem
        S = Q @ K^T * scale                    # WGMMA (SS scope)
        m_new = max(m, rowmax(S))               # per-element + warp_reduction_max
        P = exp2(scale * S - scale * m_new)    # per-element
        O *= exp2(scale * (m - m_new))         # per-element rescale
        l = l * exp2(scale * (m - m_new)) + rowsum(P)
        O += P @ V                              # WGMMA (SS scope)
        m = m_new
    O /= l                                     # normalize
    store O
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
D = 64
NUM_THREADS = 128


class FlashAttnPrefillBf16Multistage:
    """FlashAttention prefill kernel (bf16, single warpgroup)."""

    def __init__(self):
        self.q_dtype = cutlass.BFloat16
        self.kv_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.o_dtype = cutlass.BFloat16
        self.mma_shape = (64, 16, 16)
        self.qk_atom_layout = (1, 4, 1)
        self.pv_atom_layout = (1, 4, 1)

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
        r2s_tiled_copy_o: cute.TiledCopy,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        scale_log2: cutlass.Float32,
        BH: cutlass.Constexpr[int],
        M: cutlass.Constexpr[int],
        N: cutlass.Constexpr[int],
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

        gQ = cute.local_tile(mQ, (1, BLK_M, Dd), (bid_bh, bid_m, 0))
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

        scale_val = (1.0 / (Dd**0.5)) * 1.4426950408889634
        num_k_blocks = cute.size(tCrQ, mode=[2])
        num_n_blocks = (N + BLK_N - 1) // BLK_N

        for j_block in cutlass.range(num_n_blocks, unroll=1):
            gK = cute.local_tile(mK, (1, BLK_N, Dd), (bid_bh, j_block, 0))
            cute.autovec_copy(cute.group_modes(gK, 0, 2), sK)
            gV = cute.local_tile(mV, (1, Dd, BLK_N), (bid_bh, 0, j_block))
            cute.autovec_copy(cute.group_modes(gV, 1, 3), sV)
            cute.arch.sync_threads()

            tCsK = qk_thr.partition_B(sK)
            tCrK = qk_tiled_mma.make_fragment_B(tCsK)

            cute.nvgpu.warpgroup.fence()
            qk_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            for k_idx in cutlass.range(num_k_blocks, unroll_full=True):
                cute.gemm(qk_tiled_mma, tCrS, tCrQ[None, None, k_idx], tCrK[None, None, k_idx], tCrS)
                qk_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            for i in cutlass.range_constexpr(qk_m):
                s_max_prev[i] = s_max[i]
                for j in cutlass.range_constexpr(qk_n):
                    s_max[i] = cutlass.max(s_max[i], acc_qk_mn[i, j])
                for r in cutlass.range_constexpr(red_rank):
                    s_max[i] = cute.arch.warp_reduction_max(s_max[i], threads_in_group=red_target.shape[r])

                local_max = s_max[i]
                if s_max[i] == float("-inf"):
                    local_max = 0.0

                for j in cutlass.range_constexpr(qk_n):
                    acc_qk_mn[i, j] = cute.math.exp2(
                        scale_val * acc_qk_mn[i, j] - scale_val * local_max,
                        fastmath=True,
                    )

                scale_pv = cute.math.exp2(
                    (s_max_prev[i] - local_max) * scale_val,
                    fastmath=True,
                )
                a_sum[i] = a_sum[i] * scale_pv
                for j in cutlass.range_constexpr(pv_n):
                    acc_pv_mn[i, j] = acc_pv_mn[i, j] * scale_pv

                a_sum[i] = a_sum[i] + acc_qk_mn[i, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0)
                for r in cutlass.range_constexpr(red_rank):
                    a_sum[i] = cute.arch.warp_reduction_sum(a_sum[i], threads_in_group=red_target.shape[r])

            acc_qk_fixed = self.make_acc_into_op(tCrS, pv_tiled_mma.tv_layout_A, bf16)
            tCsV = pv_thr.partition_B(sV)
            tCrV = pv_tiled_mma.make_fragment_B(tCsV)

            cute.nvgpu.warpgroup.fence()
            pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.gemm(pv_tiled_mma, tCrO, acc_qk_fixed, tCrV, tCrO)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            cute.arch.sync_threads()

        red_target_pv = self.reduction_target_n(pv_tiled_mma)
        red_rank_pv = cute.rank(red_target_pv)
        for i in cutlass.range_constexpr(qk_m):
            for r in cutlass.range_constexpr(red_rank_pv):
                a_sum[i] = cute.arch.warp_reduction_sum(a_sum[i], threads_in_group=red_target_pv.shape[r])
            s = a_sum[i]
            inv = cute.arch.rcp_approx(s)
            if s == 0.0 or s != s:  # noqa: PLR0124
                inv = 1.0
            for j in cutlass.range_constexpr(pv_n):
                acc_pv_mn[i, j] = acc_pv_mn[i, j] * inv

        gO = cute.local_tile(mO, (1, BLK_M, Dd), (bid_bh, bid_m, 0))
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
        stream: CUstream,
        BH: cutlass.Constexpr[int],
        M: cutlass.Constexpr[int],
        N: cutlass.Constexpr[int],
        Dd: cutlass.Constexpr[int],
    ):
        bf16 = self.q_dtype
        f32 = self.acc_dtype
        op_qk = cute.nvgpu.warpgroup.MmaF16BF16Op(
            bf16, f32, self.mma_shape, OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K
        )
        qk_mma = cute.make_tiled_mma(op_qk, atom_layout_mnk=self.qk_atom_layout)
        op_pv = cute.nvgpu.warpgroup.MmaF16BF16Op(
            bf16, f32, self.mma_shape, OperandSource.RMEM, OperandMajorMode.K, OperandMajorMode.K
        )
        pv_mma = cute.make_tiled_mma(op_pv, atom_layout_mnk=self.pv_atom_layout)

        universal = cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(universal, bf16)
        r2s_o = cute.make_tiled_copy_C(copy_atom, pv_mma)

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

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sV_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sP_layout)], 1024]
            sO: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sO_layout)], 1024]

        scale_log2 = cutlass.Float32((1.0 / (Dd**0.5)) * 1.4426950408889634)
        grid = (BH, (M + BLK_M - 1) // BLK_M, 1)

        self.kernel(
            qk_mma,
            pv_mma,
            mQ,
            mK,
            mV,
            mO,
            r2s_o,
            sQ_layout,
            sK_layout,
            sV_layout,
            sP_layout,
            sO_layout,
            scale_log2,
            BH,
            M,
            N,
            Dd,
            SharedStorage,
        ).launch(grid=grid, block=(NUM_THREADS, 1, 1), stream=stream)


__all__ = ["BLK_M", "BLK_N", "NUM_THREADS", "D", "FlashAttnPrefillBf16Multistage"]
