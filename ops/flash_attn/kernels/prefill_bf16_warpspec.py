"""FlashAttention prefill — bf16, warp-specialized (Exercise 2).

Maps to hpc-ops ``A2 (warp_spec_dim128)``.

- Warp specialization: 384 threads = 1 producer WG (TMA) + 2 consumer WGs (WGMMA).
- Persistent grid, ``kStage=2``, ``TileM=128``.
- bf16, ``head_dim=128``.

Same two-pass + V^T layout strategy as exercise 1, but with:
- Producer/consumer WG split (``warpgroup_reg_alloc/dealloc``)
- ``PipelineTmaAsync`` for K/V multi-stage loads
- Persistent grid stride loop
"""

from __future__ import annotations

import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
from cuda.bindings import driver as cuda_driver
from cuda.bindings.driver import CUstream
from cutlass import cute, pipeline
from cutlass.cute.nvgpu.warpgroup import OperandMajorMode, OperandSource
from cutlass.utils.layout import LayoutEnum


BLK_M = 128
BLK_N = 64
D = 128
NUM_STAGES = 2

NUM_DMA_WAROGROUPS = 1
NUM_MMA_WAROGROUPS = 2
NUM_WAROGROUPS = NUM_DMA_WAROGROUPS + NUM_MMA_WAROGROUPS
NUM_THREADS_PER_WAROGROUP = 128
NUM_THREADS = NUM_WAROGROUPS * NUM_THREADS_PER_WAROGROUP
NUM_WARPS = NUM_THREADS // 32
NUM_MMA_THREADS = NUM_MMA_WAROGROUPS * NUM_THREADS_PER_WAROGROUP
MMA_NAMED_BARRIER_ID = 1
LOAD_REGISTER_REQUIREMENT = 24
MMA_REGISTER_REQUIREMENT = 232


@cute.kernel
def flash_attn_prefill_bf16_warpspec_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    tma_atom_v: cute.CopyAtom,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    r2s_tiled_copy_o: cute.TiledCopy,
    r2s_tiled_copy_p: cute.TiledCopy,
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sP_layout: cute.ComposedLayout,
    sO_layout: cute.ComposedLayout,
    is_causal: cutlass.Constexpr[bool],
    BH: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    Dd: cutlass.Constexpr[int],
    num_persistent: cutlass.Constexpr[int],
    tx_count_k: cutlass.Constexpr[int],
    tx_count_v: cutlass.Constexpr[int],
    shared_storage_cls: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warpgroup_idx = cute.arch.make_warp_uniform(tid // NUM_THREADS_PER_WAROGROUP)
    is_producer = warpgroup_idx < NUM_DMA_WAROGROUPS

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage_cls)

    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK_full = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV_full = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
    sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

    mbar_ptr = storage.mbar.data_ptr()

    grid_m = (M + BLK_M - 1) // BLK_M
    total_tiles = BH * grid_m
    tiles_per_cta = (total_tiles + num_persistent - 1) // num_persistent

    num_n_blocks = (N + BLK_N - 1) // BLK_N

    sK_for_tma = cute.group_modes(sK_full, 0, 2)
    sV_for_tma = cute.group_modes(sV_full, 0, 2)

    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=mbar_ptr,
        num_stages=NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_WARPS),
        tx_count=tx_count_k + tx_count_v,
    )
    prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_STAGES)
    cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NUM_STAGES)

    if is_producer:
        cute.arch.warpgroup_reg_dealloc(LOAD_REGISTER_REQUIREMENT)

        for tile_iter in cutlass.range(tiles_per_cta, unroll=1):
            tile_idx = tile_iter * num_persistent + bidx
            if tile_idx < total_tiles:
                bid_bh = tile_idx // grid_m
                bid_m = tile_idx % grid_m
                causal_limit = num_n_blocks
                if is_causal:
                    causal_limit = min(num_n_blocks, ((bid_m + 1) * BLK_M + BLK_N - 1) // BLK_N)

                gQ = cute.local_tile(mQ, (1, BLK_M, Dd), (bid_bh, bid_m, 0))
                gQ_2d = cute.group_modes(gQ, 0, 2)
                cute.autovec_copy(gQ_2d, sQ)

                for j_block in cutlass.range(causal_limit, unroll=1):
                    gK = cute.local_tile(mK, (1, BLK_N, Dd), (bid_bh, j_block, 0))
                    gK_2d = cute.group_modes(gK, 0, 2)
                    gV = cute.local_tile(mV, (1, Dd, BLK_N), (bid_bh, 0, j_block))
                    gV_2d = cute.group_modes(gV, 0, 2)

                    tAsK, tAgK = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k, 0, cute.make_layout(1), sK_for_tma, gK_2d
                    )
                    tAsV, tAgV = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v, 0, cute.make_layout(1), sV_for_tma, gV_2d
                    )

                    mainloop_pipeline.producer_acquire(prod_state)
                    cute.copy(
                        tma_atom_k,
                        tAgK[None, 0],
                        tAsK[None, prod_state.index],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                    cute.copy(
                        tma_atom_v,
                        tAgV[None, 0],
                        tAsV[None, prod_state.index],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()
    else:
        cute.arch.warpgroup_reg_alloc(MMA_REGISTER_REQUIREMENT)

        mma_wg_idx = warpgroup_idx - NUM_DMA_WAROGROUPS
        wg_layout = cute.make_layout(NUM_MMA_WAROGROUPS, stride=NUM_THREADS_PER_WAROGROUP)
        thr_qk = tiled_mma_qk.get_slice(wg_layout(mma_wg_idx))
        thr_pv = tiled_mma_pv.get_slice(wg_layout(mma_wg_idx))

        tCsQ = thr_qk.partition_A(sQ)
        tCrQ = tiled_mma_qk.make_fragment_A(tCsQ)
        tCsS = thr_qk.partition_C(sP)
        tCrS = tiled_mma_qk.make_fragment_C(tCsS)

        tCsP_a = thr_pv.partition_A(sP)
        tCrP_a = tiled_mma_pv.make_fragment_A(tCsP_a)
        tCsO = thr_pv.partition_C(sO)
        tCrO = tiled_mma_pv.make_fragment_C(tCsO)

        thr_r2s_p = r2s_tiled_copy_p.get_slice(tid)
        tDsP_r2s = thr_r2s_p.partition_D(sP)
        thr_r2s_o = r2s_tiled_copy_o.get_slice(tid)

        tCrO.fill(0.0)
        acc_c_shape = tCsS.shape
        reduced_shape = (acc_c_shape[0], acc_c_shape[1])
        m_i = cute.make_rmem_tensor(reduced_shape, cutlass.Float32)
        l_i = cute.make_rmem_tensor(reduced_shape, cutlass.Float32)
        m_i.fill(float("-inf"))
        l_i.fill(0.0)
        scale_tensor = cute.make_rmem_tensor(acc_c_shape, cutlass.Float32)
        scale_tensor.fill(1.0 / (Dd**0.5))

        num_k_blocks_qk = cute.size(tCrQ, mode=[2])
        num_k_blocks_pv = cute.size(tCrP_a, mode=[2])

        # Pass 1: compute m_global + l_global (QK only)
        for tile_iter in cutlass.range(tiles_per_cta, unroll=1):
            tile_idx = tile_iter * num_persistent + bidx
            if tile_idx < total_tiles:
                bid_bh = tile_idx // grid_m
                bid_m = tile_idx % grid_m
                causal_limit = num_n_blocks
                if is_causal:
                    causal_limit = min(num_n_blocks, ((bid_m + 1) * BLK_M + BLK_N - 1) // BLK_N)

                m_i.fill(float("-inf"))
                l_i.fill(0.0)

                for j_block in cutlass.range(causal_limit, unroll=1):
                    mainloop_pipeline.consumer_wait(cons_state)
                    sK_st = sK_full[None, cons_state.index]

                    tCsK = thr_qk.partition_B(sK_st)
                    tCrK = tiled_mma_qk.make_fragment_B(tCsK)

                    cute.nvgpu.warpgroup.fence()
                    tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                    for k_idx in cutlass.range(num_k_blocks_qk, unroll_full=True):
                        coord = (None, None, k_idx)
                        cute.gemm(tiled_mma_qk, tCrS, tCrQ[coord], tCrK[coord], tCrS)
                        tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    s_val = cute.math.mul(tCrS.load(), scale_tensor.load())
                    m_new = s_val.reduce(cute.ReductionOp.MAX, float("-inf"), (None, None, 1))
                    m_new_b = m_new.broadcast_to(acc_c_shape)
                    p_val = cute.math.exp(cute.math.sub(s_val, m_new_b))
                    t_val = p_val.reduce(cute.ReductionOp.ADD, 0.0, (None, None, 1))

                    alpha = cute.math.exp(cute.math.sub(m_i.load(), m_new))
                    l_i.store(cute.math.add(cute.math.mul(l_i.load(), alpha), t_val))
                    m_i.store(m_new)

                    mainloop_pipeline.consumer_release(cons_state)
                    cons_state.advance()

                # Pass 2: O = sum(P_norm @ V)
                tCrO.fill(0.0)
                # Reset consumer state to re-read the same K/V tiles
                # In practice, producer must re-issue loads for pass 2.
                # For simplicity, we use autovec_copy for K and TMA for V in pass 2.

                for j_block in cutlass.range(causal_limit, unroll=1):
                    # Re-load K via autovec (simpler than re-pipelining)
                    gK = cute.local_tile(mK, (1, BLK_N, Dd), (bid_bh, j_block, 0))
                    gK_2d = cute.group_modes(gK, 0, 2)
                    cute.autovec_copy(gK_2d, sK_st)
                    cute.arch.sync_threads()

                    # Load V via TMA
                    gV = cute.local_tile(mV, (1, Dd, BLK_N), (bid_bh, 0, j_block))
                    gV_2d = cute.group_modes(gV, 0, 2)
                    tAsV, tAgV = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v, 0, cute.make_layout(1), sV_for_tma, gV_2d
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(mbar_ptr + 8, 1)
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + 8, tx_count_v)
                    cute.copy(tma_atom_v, tAgV[None, 0], tAsV[None, 0], tma_bar_ptr=mbar_ptr + 8)
                    cute.arch.mbarrier_wait(mbar_ptr + 8, 0)
                    cute.arch.sync_threads()

                    sV_st = sV_full[None, 0]
                    tCsV = thr_pv.partition_B(sV_st)
                    tCrV = tiled_mma_pv.make_fragment_B(tCsV)

                    # QK matmul
                    cute.nvgpu.warpgroup.fence()
                    tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                    for k_idx in cutlass.range(num_k_blocks_qk, unroll_full=True):
                        coord = (None, None, k_idx)
                        cute.gemm(tiled_mma_qk, tCrS, tCrQ[coord], tCrK[coord], tCrS)
                        tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    # Softmax with global m, l (pre-normalized P)
                    s_val = cute.math.mul(tCrS.load(), scale_tensor.load())
                    m_global_b = m_i.load().broadcast_to(acc_c_shape)
                    l_global_b = l_i.load().broadcast_to(acc_c_shape)
                    p_val = cute.math.exp(cute.math.sub(s_val, m_global_b))
                    p_val = cute.math.div(p_val, l_global_b)
                    tCrS.store(p_val)

                    # r2s: store P to sP
                    tCrS_bf16 = cute.make_fragment_like(tCrS, cutlass.BFloat16)
                    tCrS_bf16.store(tCrS.load().to(cutlass.BFloat16))
                    tDrS_r2s = thr_r2s_p.retile(tCrS_bf16)
                    cute.copy(r2s_tiled_copy_p, tDrS_r2s, tDsP_r2s)
                    cute.arch.sync_threads()

                    # PV matmul
                    cute.nvgpu.warpgroup.fence()
                    for k_idx in cutlass.range(num_k_blocks_pv, unroll_full=True):
                        coord = (None, None, k_idx)
                        cute.gemm(tiled_mma_pv, tCrO, tCrP_a[coord], tCrV[coord], tCrO)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                    cute.arch.sync_threads()

                # Epilogue: store O to gmem
                tCrO_bf16 = cute.make_fragment_like(tCrO, cutlass.BFloat16)
                tCrO_bf16.store(tCrO.load().to(cutlass.BFloat16))
                tDrO = thr_r2s_o.retile(tCrO_bf16)
                gO = cute.local_tile(mO, (1, BLK_M, Dd), (bid_bh, bid_m, 0))
                gO_2d = cute.group_modes(gO, 0, 2)
                tDgO = thr_r2s_o.partition_D(gO_2d)
                cute.copy(r2s_tiled_copy_o, tDrO, tDgO)


@cute.jit
def flash_attn_prefill_bf16_warpspec(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    stream: CUstream,
    is_causal: cutlass.Constexpr[bool],
    BH: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    Dd: cutlass.Constexpr[int],
):
    bf16 = cutlass.BFloat16
    f32 = cutlass.Float32
    mma_shape = (64, 16, 16)

    op_qk = cute.nvgpu.warpgroup.MmaF16BF16Op(
        bf16, f32, mma_shape, OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K
    )
    tiled_mma_qk = cute.make_tiled_mma(cute.make_mma_atom(op_qk), (2, 4, 1))

    op_pv = cute.nvgpu.warpgroup.MmaF16BF16Op(
        bf16, f32, mma_shape, OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K
    )
    tiled_mma_pv = cute.make_tiled_mma(cute.make_mma_atom(op_pv), (2, 8, 1))

    universal = cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(universal, bf16)
    r2s_tiled_copy_o = cute.make_tiled_copy_C(copy_atom, tiled_mma_pv)
    r2s_tiled_copy_p = cute.make_tiled_copy_C(copy_atom, tiled_mma_qk)

    q_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, Dd), bf16)
    k_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, Dd), bf16)
    v_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, BLK_N), bf16)
    p_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, BLK_N), bf16)

    sQ_layout = cute.tile_to_shape(q_atom, (BLK_M, Dd), order=(0, 1))
    sK_layout = cute.tile_to_shape(k_atom, (BLK_N, Dd, NUM_STAGES), order=(0, 1, 2))
    sV_layout = cute.tile_to_shape(v_atom, (Dd, BLK_N, NUM_STAGES), order=(0, 1, 2))
    sP_layout = cute.tile_to_shape(p_atom, (BLK_M, BLK_N), order=(0, 1))
    sO_layout = cute.tile_to_shape(q_atom, (BLK_M, Dd), order=(0, 1))

    sK_layout_one = cute.slice_(sK_layout, (None, None, 0))
    sV_layout_one = cute.slice_(sV_layout, (None, None, 0))

    tma_atom_k, _ = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mK, sK_layout_one, (BLK_N, Dd)
    )
    tma_atom_v, _ = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mV, sV_layout_one, (Dd, BLK_N)
    )

    tx_count_k = BLK_N * Dd * 2
    tx_count_v = Dd * BLK_N * 2

    @cute.struct
    class SharedStorage:
        sQ: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sQ_layout)], 1024]
        sK: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sK_layout)], 1024]
        sV: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sV_layout)], 1024]
        sP: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sP_layout)], 1024]
        sO: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sO_layout)], 1024]
        mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 8]

    grid_m = (M + BLK_M - 1) // BLK_M
    total_tiles = BH * grid_m
    num_sms = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0
    )[1]
    num_persistent = min(total_tiles, num_sms)

    flash_attn_prefill_bf16_warpspec_kernel(
        mQ,
        mK,
        mV,
        mO,
        tma_atom_k,
        tma_atom_v,
        tiled_mma_qk,
        tiled_mma_pv,
        r2s_tiled_copy_o,
        r2s_tiled_copy_p,
        sQ_layout,
        sK_layout,
        sV_layout,
        sP_layout,
        sO_layout,
        is_causal,
        BH,
        M,
        N,
        Dd,
        num_persistent,
        tx_count_k,
        tx_count_v,
        SharedStorage,
    ).launch(grid=(num_persistent, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


__all__ = [
    "BLK_M",
    "BLK_N",
    "LOAD_REGISTER_REQUIREMENT",
    "MMA_NAMED_BARRIER_ID",
    "MMA_REGISTER_REQUIREMENT",
    "NUM_DMA_WAROGROUPS",
    "NUM_MMA_THREADS",
    "NUM_MMA_WAROGROUPS",
    "NUM_STAGES",
    "NUM_THREADS",
    "NUM_THREADS_PER_WAROGROUP",
    "NUM_WAROGROUPS",
    "NUM_WARPS",
    "D",
    "flash_attn_prefill_bf16_warpspec",
    "flash_attn_prefill_bf16_warpspec_kernel",
]
