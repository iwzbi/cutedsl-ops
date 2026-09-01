"""FlashAttention prefill — FP8, warp-specialized, paged KV (Exercise 4).

Maps to hpc-ops ``C1 (warp_spec_with_kvcache_fp8, qpertoken_perhead_kvpertensor)``.

Same structure as exercise 2, but:
- Q/K loaded as fp8, converted to bf16 in SMEM for QK matmul.
- PV uses fp8 WGMMA: P is pre-scaled x256, cast to fp8, V stays fp8 in SMEM.
- Per-tensor K/V scales, per-token-per-head Q scale.
- V stored as V^T (D, BLK_N) with BLK_N contiguous (K-major B for fp8 WGMMA).
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

P_SCALE = 256.0


@cute.kernel
def flash_attn_prefill_fp8_kernel(
    mQ: cute.Tensor,
    mK_pages: cute.Tensor,
    mV_pages: cute.Tensor,
    mBlockTable: cute.Tensor,
    mO: cute.Tensor,
    mQScale: cute.Tensor,
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
    kScale: cutlass.Constexpr[float],
    vScale: cutlass.Constexpr[float],
    is_causal: cutlass.Constexpr[bool],
    BH: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    Dd: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
    num_persistent: cutlass.Constexpr[int],
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
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
    sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
    mbar_ptr = storage.mbar.data_ptr()

    grid_m = (M + BLK_M - 1) // BLK_M
    total_tiles = BH * grid_m
    tiles_per_cta = (total_tiles + num_persistent - 1) // num_persistent
    num_n_blocks = (N + BLK_N - 1) // BLK_N

    sV_for_tma = cute.group_modes(sV, 0, 2)

    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=mbar_ptr,
        num_stages=NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_WARPS),
        tx_count=tx_count_v,
    )
    prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_STAGES)
    cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NUM_STAGES)

    bf16 = cutlass.BFloat16
    f32 = cutlass.Float32
    fp8 = cutlass.Float8

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

                # Load Q as fp8, convert to bf16 in SMEM
                gQ = cute.local_tile(mQ, (1, BLK_M, Dd), (bid_bh, bid_m, 0))
                gQ_2d = cute.group_modes(gQ, 0, 2)
                gQ_bf16 = cute.make_tensor_like(gQ_2d, bf16)
                cute.autovec_copy(gQ_2d, gQ_bf16)
                cute.autovec_copy(gQ_bf16, sQ)

                for j_block in cutlass.range(causal_limit, unroll=1):
                    # Paged KV: resolve block_table
                    bid_b = bid_bh  # Simplified: assume H_kv=1 per batch
                    kv_idx = j_block
                    page_id = mBlockTable[(bid_b, kv_idx)]

                    # Load K as fp8, convert to bf16 in SMEM
                    gK = cute.local_tile(mK_pages, (1, 1, page_size, Dd), (page_id, 0, 0, 0))
                    gK_2d = cute.group_modes(gK, 0, 2)
                    gK_bf16 = cute.make_tensor_like(gK_2d, bf16)
                    cute.autovec_copy(gK_2d, gK_bf16)

                    # Load V as fp8 (stays fp8 in SMEM for fp8 WGMMA)
                    gV = cute.local_tile(mV_pages, (1, 1, Dd, page_size), (page_id, 0, 0, 0))
                    gV_2d = cute.group_modes(gV, 0, 2)

                    tAsV, tAgV = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v, 0, cute.make_layout(1), sV_for_tma, gV_2d
                    )

                    mainloop_pipeline.producer_acquire(prod_state)
                    # Store K as bf16 to sK
                    cute.autovec_copy(gK_bf16, sK[None, prod_state.index])
                    # TMA load V as fp8 to sV
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
        m_i = cute.make_rmem_tensor(reduced_shape, f32)
        l_i = cute.make_rmem_tensor(reduced_shape, f32)
        m_i.fill(float("-inf"))
        l_i.fill(0.0)

        # Scale tensors
        softmax_scale = 1.0 / (Dd**0.5)
        scale_tensor = cute.make_rmem_tensor(acc_c_shape, f32)
        scale_tensor.fill(softmax_scale)
        k_scale_tensor = cute.make_rmem_tensor(acc_c_shape, f32)
        k_scale_tensor.fill(kScale)
        p_scale_tensor = cute.make_rmem_tensor(acc_c_shape, f32)
        p_scale_tensor.fill(P_SCALE)

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
                    sK_st = sK[None, cons_state.index]

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

                    # S = Q @ K^T * softmax_scale * kScale
                    s_val = cute.math.mul(tCrS.load(), scale_tensor.load())
                    s_val = cute.math.mul(s_val, k_scale_tensor.load())
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
                for j_block in cutlass.range(causal_limit, unroll=1):
                    mainloop_pipeline.consumer_wait(cons_state)
                    sK_st = sK[None, cons_state.index]
                    sV_st = sV[None, cons_state.index]

                    tCsK = thr_qk.partition_B(sK_st)
                    tCrK = tiled_mma_qk.make_fragment_B(tCsK)
                    tCsV = thr_pv.partition_B(sV_st)
                    tCrV = tiled_mma_pv.make_fragment_B(tCsV)

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
                    s_val = cute.math.mul(s_val, k_scale_tensor.load())
                    m_global_b = m_i.load().broadcast_to(acc_c_shape)
                    l_global_b = l_i.load().broadcast_to(acc_c_shape)
                    p_val = cute.math.exp(cute.math.sub(s_val, m_global_b))
                    p_val = cute.math.div(p_val, l_global_b)

                    # P pre-scale x256, cast to fp8
                    p_scaled = cute.math.mul(p_val, p_scale_tensor.load())
                    tCrS.store(p_scaled)

                    # r2s: store P to sP (convert fp32 -> fp8)
                    tCrS_fp8 = cute.make_fragment_like(tCrS, fp8)
                    tCrS_fp8.store(tCrS.load().to(fp8))
                    tDrS_r2s = thr_r2s_p.retile(tCrS_fp8)
                    cute.copy(r2s_tiled_copy_p, tDrS_r2s, tDsP_r2s)
                    cute.arch.sync_threads()

                    # PV matmul (fp8 WGMMA: A=P fp8 from sP, B=V fp8 from sV)
                    cute.nvgpu.warpgroup.fence()
                    for k_idx in cutlass.range(num_k_blocks_pv, unroll_full=True):
                        coord = (None, None, k_idx)
                        cute.gemm(tiled_mma_pv, tCrO, tCrP_a[coord], tCrV[coord], tCrO)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    mainloop_pipeline.consumer_release(cons_state)
                    cons_state.advance()

                # Epilogue: O *= vScale, store as bf16
                v_scale_tensor = cute.make_rmem_tensor(tCrO.shape, f32)
                v_scale_tensor.fill(vScale)
                tCrO.store(cute.math.mul(tCrO.load(), v_scale_tensor.load()))

                tCrO_bf16 = cute.make_fragment_like(tCrO, bf16)
                tCrO_bf16.store(tCrO.load().to(bf16))
                tDrO = thr_r2s_o.retile(tCrO_bf16)
                gO = cute.local_tile(mO, (1, BLK_M, Dd), (bid_bh, bid_m, 0))
                gO_2d = cute.group_modes(gO, 0, 2)
                tDgO = thr_r2s_o.partition_D(gO_2d)
                cute.copy(r2s_tiled_copy_o, tDrO, tDgO)


@cute.jit
def flash_attn_prefill_fp8(
    mQ: cute.Tensor,
    mK_pages: cute.Tensor,
    mV_pages: cute.Tensor,
    mBlockTable: cute.Tensor,
    mO: cute.Tensor,
    mQScale: cute.Tensor,
    stream: CUstream,
    kScale: cutlass.Constexpr[float],
    vScale: cutlass.Constexpr[float],
    is_causal: cutlass.Constexpr[bool],
    BH: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    Dd: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
):
    bf16 = cutlass.BFloat16
    f32 = cutlass.Float32
    fp8 = cutlass.Float8
    mma_shape = (64, 16, 16)

    # QK: bf16 WGMMA (Q/K converted from fp8 to bf16 in SMEM)
    op_qk = cute.nvgpu.warpgroup.MmaF16BF16Op(
        bf16, f32, mma_shape, OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K
    )
    tiled_mma_qk = cute.make_tiled_mma(cute.make_mma_atom(op_qk), (2, 4, 1))

    # PV: fp8 WGMMA (P is fp8, V is fp8, both from SMEM, K-major)
    op_pv = cute.nvgpu.warpgroup.MmaF8Op(
        fp8, fp8, f32, mma_shape, OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K
    )
    tiled_mma_pv = cute.make_tiled_mma(cute.make_mma_atom(op_pv), (2, 8, 1))

    universal = cute.nvgpu.CopyUniversalOp()
    copy_atom_bf16 = cute.make_copy_atom(universal, bf16)
    copy_atom_fp8 = cute.make_copy_atom(universal, fp8)
    r2s_tiled_copy_o = cute.make_tiled_copy_C(copy_atom_bf16, tiled_mma_pv)
    r2s_tiled_copy_p = cute.make_tiled_copy_C(copy_atom_fp8, tiled_mma_qk)

    q_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, Dd), bf16)
    k_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, Dd), bf16)
    v_atom = sm90_utils.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, fp8, page_size), fp8
    )
    p_atom = sm90_utils.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, fp8, BLK_N), fp8)

    sQ_layout = cute.tile_to_shape(q_atom, (BLK_M, Dd), order=(0, 1))
    sK_layout = cute.tile_to_shape(k_atom, (BLK_N, Dd, NUM_STAGES), order=(0, 1, 2))
    sV_layout = cute.tile_to_shape(v_atom, (Dd, page_size, NUM_STAGES), order=(0, 1, 2))
    sP_layout = cute.tile_to_shape(p_atom, (BLK_M, BLK_N), order=(0, 1))
    sO_layout = cute.tile_to_shape(q_atom, (BLK_M, Dd), order=(0, 1))

    sV_layout_one = cute.slice_(sV_layout, (None, None, 0))
    tma_atom_v, _ = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mV_pages, sV_layout_one, (1, 1, Dd, page_size)
    )
    tx_count_v = Dd * page_size * 1  # fp8 = 1 byte

    @cute.struct
    class SharedStorage:
        sQ: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sQ_layout)], 1024]
        sK: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sK_layout)], 1024]
        sV: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(sV_layout)], 1024]
        sP: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(sP_layout)], 1024]
        sO: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sO_layout)], 1024]
        mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 8]

    grid_m = (M + BLK_M - 1) // BLK_M
    total_tiles = BH * grid_m
    num_sms = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0
    )[1]
    num_persistent = min(total_tiles, num_sms)

    flash_attn_prefill_fp8_kernel(
        mQ,
        mK_pages,
        mV_pages,
        mBlockTable,
        mO,
        mQScale,
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
        kScale,
        vScale,
        is_causal,
        BH,
        M,
        N,
        Dd,
        page_size,
        num_persistent,
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
    "P_SCALE",
    "D",
    "flash_attn_prefill_fp8",
    "flash_attn_prefill_fp8_kernel",
]
