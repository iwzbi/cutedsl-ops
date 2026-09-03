"""Cluster GEMM kernel — TMA multicast + cluster (2,1), no warp specialization.

Based on the official CuTe DSL dense_gemm.py pattern:
- 2 warpgroups (256 threads), warp0 issues TMA + all warps do WGMMA
- Cluster (2,1): B multicast along M (2 CTAs share same B tile)
- make_layout_image_mask for compile-time multicast masks
- pipeline_init_arrive/wait for cluster barrier setup
- defer_sync=True + external consumer_arrive_cnt (no enable_multicast_signaling)
- Prefetch + prologue MMA + mainloop (consumer_try_wait peek pattern)
"""

from __future__ import annotations

import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
from cuda.bindings.driver import CUstream
from cutlass import cute, pipeline
from cutlass.cute.nvgpu.warpgroup import OperandMajorMode, OperandSource
from cutlass.utils.layout import LayoutEnum


BLK_M = 128
BLK_N = 128
BLK_K = 64
NUM_STAGES = 3
CLUSTER_M = 2
CLUSTER_N = 1
CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

ATOM_LAYOUT_MNK = (2, 1, 1)
NUM_MMA_WARPGROUPS = ATOM_LAYOUT_MNK[0] * ATOM_LAYOUT_MNK[1] * ATOM_LAYOUT_MNK[2]
NUM_WARPGROUPS = NUM_MMA_WARPGROUPS  # no DMA warpgroup
NUM_THREADS_PER_WARPGROUP = 128
NUM_THREADS = NUM_WARPGROUPS * NUM_THREADS_PER_WARPGROUP
NUM_WARPS = NUM_THREADS // 32


@cute.kernel
def gemm_kernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_atom_d: cute.CopyAtom,
    mD: cute.Tensor,
    tiled_mma: cute.TiledMma,
    r2s_tiled_copy_d: cute.TiledCopy,
    sA_layout_staged: cute.ComposedLayout,
    sB_layout_staged: cute.ComposedLayout,
    sD_layout: cute.ComposedLayout,
    tx_count_ab: cutlass.Constexpr,
    cta_layout_vmnk: cute.Layout,
    cta_layout_mnk: cute.Layout,
    grid_m: cutlass.Constexpr[int],
    grid_n: cutlass.Constexpr[int],
    acc_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    shared_storage_cls: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage_cls)
    sA_full = storage.sA.get_tensor(sA_layout_staged.outer, swizzle=sA_layout_staged.inner)
    sB_full = storage.sB.get_tensor(sB_layout_staged.outer, swizzle=sB_layout_staged.inner)
    sD = storage.sD.get_tensor(sD_layout.outer, swizzle=sD_layout.inner)

    # --- Cluster position + multicast masks ---
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
    a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
    b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)
    # Static (layout-derived) masks; gated to 0 for non-multicast dims so the
    # plain-atom copies stay legal. Lesson-12 shape — runtime-conditional
    # masks hang MLIR (old PERFLOG #17), do not introduce them.
    a_mcast_mask = a_mcast_mask if CLUSTER_N > 1 else cutlass.Int16(0)
    b_mcast_mask = b_mcast_mask if CLUSTER_M > 1 else cutlass.Int16(0)

    # --- Pipeline with cluster ---
    mcast_size = CLUSTER_M + CLUSTER_N - 1
    consumer_arrive_cnt = mcast_size * NUM_WARPS
    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mainloop_mbar_array.data_ptr(),
        num_stages=NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt),
        tx_count=tx_count_ab,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    )
    pipeline.pipeline_init_arrive(cluster_shape_mn=(CLUSTER_M, CLUSTER_N), is_relaxed=True)

    # --- MMA setup (all warps are consumers) ---
    warpgroup_idx = cute.arch.make_warp_uniform(tid // NUM_THREADS_PER_WARPGROUP)
    warpgroup_thread_layout = cute.make_layout(NUM_MMA_WARPGROUPS, stride=NUM_THREADS_PER_WARPGROUP)
    thr_mma = tiled_mma.get_slice(warpgroup_thread_layout(warpgroup_idx))
    tCsA = thr_mma.partition_A(sA_full)
    tCsB = thr_mma.partition_B(sB_full)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    num_k_blocks = cute.size(tCrA, mode=[2])

    # --- local_tile + TMA partition ---
    tiler = (BLK_M, BLK_N, BLK_K)
    gA = cute.local_tile(mA, tiler=tiler, coord=(bidy, bidx, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=tiler, coord=(bidy, bidx, None), proj=(None, 1, 1))
    gD = cute.local_tile(mD, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))

    sA_for_tma = cute.group_modes(sA_full, 0, 2)
    gA_for_tma = cute.group_modes(gA, 0, 2)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        sA_for_tma,
        gA_for_tma,
    )
    sB_for_tma = cute.group_modes(sB_full, 0, 2)
    gB_for_tma = cute.group_modes(gB, 0, 2)
    if cutlass.const_expr(CLUSTER_M > 1):
        # B tiles are shared across cluster-mode-0 (M): partition with the
        # per-axis cta coord/layout so each CTA issues only its 1/CLUSTER_M
        # slice; the multicast mask fans every slice into ALL members' sB.
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape,
        )
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sB_for_tma,
            gB_for_tma,
        )
    else:
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            sB_for_tma,
            gB_for_tma,
        )
    sD_for_tma = cute.group_modes(
        cute.make_tensor(sD.iterator, cute.append(sD.layout, cute.make_layout(1))),
        0,
        2,
    )
    gD_for_tma = cute.group_modes(
        cute.make_tensor(gD.iterator, cute.append(gD.layout, cute.make_layout(1))),
        0,
        2,
    )
    tDsD, tDgD = cute.nvgpu.cpasync.tma_partition(
        tma_atom_d,
        0,
        cute.make_layout(1),
        sD_for_tma,
        gD_for_tma,
    )

    k_tile_cnt = cute.size(gA, mode=[2])

    # --- Accumulator ---
    tCgD_shape = thr_mma.partition_C(gD).shape
    accumulators = cute.make_rmem_tensor(tCgD_shape, acc_dtype)
    accumulators.fill(0.0)

    # --- Pipeline states ---
    mainloop_prod_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        NUM_STAGES,
    )
    mainloop_cons_read_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        NUM_STAGES,
    )
    mainloop_cons_release_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        NUM_STAGES,
    )

    # --- Cluster wait (after all CTAs arrived) ---
    pipeline.pipeline_init_wait(cluster_shape_mn=(CLUSTER_M, CLUSTER_N))

    # --- Prefetch (warp0 only) ---
    prefetch_cnt = cutlass.max(cutlass.min(NUM_STAGES, k_tile_cnt), 0)
    if warp_idx == 0:
        for prefetch_idx in cutlass.range(prefetch_cnt, unroll=1):
            mainloop_pipeline.producer_acquire(mainloop_prod_state)
            cute.copy(
                tma_atom_a,
                tAgA[None, mainloop_prod_state.count],
                tAsA[None, mainloop_prod_state.index],
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
            )
            cute.copy(
                tma_atom_b,
                tBgB[None, mainloop_prod_state.count],
                tBsB[None, mainloop_prod_state.index],
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
                mcast_mask=b_mcast_mask,
            )
            mainloop_pipeline.producer_commit(mainloop_prod_state)
            mainloop_prod_state.advance()

    # --- Prologue MMA (1 iteration before mainloop) ---
    k_pipe_mmas = 1
    peek_status = cutlass.Boolean(1)
    if mainloop_cons_read_state.count < k_tile_cnt:
        peek_status = mainloop_pipeline.consumer_try_wait(mainloop_cons_read_state)

    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
    for _ in cutlass.range_constexpr(k_pipe_mmas):
        mainloop_pipeline.consumer_wait(mainloop_cons_read_state, peek_status)
        cute.nvgpu.warpgroup.fence()
        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
            coord = (None, None, k_block_idx, mainloop_cons_read_state.index)
            cute.gemm(tiled_mma, accumulators, tCrA[coord], tCrB[coord], accumulators)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.nvgpu.warpgroup.commit_group()
        mainloop_cons_read_state.advance()
        peek_status = cutlass.Boolean(1)
        if mainloop_cons_read_state.count < k_tile_cnt:
            peek_status = mainloop_pipeline.consumer_try_wait(mainloop_cons_read_state)

    # --- Mainloop ---
    for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
        mainloop_pipeline.consumer_wait(mainloop_cons_read_state, peek_status)
        cute.nvgpu.warpgroup.fence()
        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
            coord = (None, None, k_block_idx, mainloop_cons_read_state.index)
            cute.gemm(tiled_mma, accumulators, tCrA[coord], tCrB[coord], accumulators)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
        mainloop_pipeline.consumer_release(mainloop_cons_release_state)
        mainloop_cons_read_state.advance()
        mainloop_cons_release_state.advance()

        peek_status = cutlass.Boolean(1)
        if mainloop_cons_read_state.count < k_tile_cnt:
            peek_status = mainloop_pipeline.consumer_try_wait(mainloop_cons_read_state)

        # Producer: prefetch next stage
        if warp_idx == 0 and mainloop_prod_state.count < k_tile_cnt:
            mainloop_pipeline.producer_acquire(mainloop_prod_state)
            cute.copy(
                tma_atom_a,
                tAgA[None, mainloop_prod_state.count],
                tAsA[None, mainloop_prod_state.index],
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
            )
            cute.copy(
                tma_atom_b,
                tBgB[None, mainloop_prod_state.count],
                tBsB[None, mainloop_prod_state.index],
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
                mcast_mask=b_mcast_mask,
            )
            mainloop_pipeline.producer_commit(mainloop_prod_state)
            mainloop_prod_state.advance()

    # --- Epilogue ---
    cute.nvgpu.warpgroup.wait_group(0)
    tCrD = cute.make_fragment_like(accumulators, out_dtype)
    tCrD.store(accumulators.load().to(out_dtype))
    thr_r2s_d = r2s_tiled_copy_d.get_slice(tid)
    tDrD_r2s = thr_r2s_d.retile(tCrD)
    tDsD_r2s = thr_r2s_d.partition_D(sD)
    cute.copy(r2s_tiled_copy_d, tDrD_r2s, tDsD_r2s)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.copy(tma_atom_d, tDsD[None, 0], tDgD[None, 0])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=False)


@cute.jit
def gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: CUstream,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    split_k: cutlass.Constexpr[int] = 1,
):
    acc_dtype = cutlass.Float32
    out_dtype = mC.element_type

    op = cute.nvgpu.warpgroup.MmaF16BF16Op(
        mA.element_type,
        acc_dtype,
        (64, BLK_N, 16),
        OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    tm = cute.make_tiled_mma(cute.make_mma_atom(op), ATOM_LAYOUT_MNK)

    a_atom = sm90_utils.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, mA.element_type, BLK_K),
        mA.element_type,
    )
    b_atom = sm90_utils.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, mB.element_type, BLK_K),
        mB.element_type,
    )
    d_atom = sm90_utils.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, out_dtype, BLK_N),
        out_dtype,
    )
    sA_layout_staged = cute.tile_to_shape(a_atom, (BLK_M, BLK_K, NUM_STAGES), order=(0, 1, 2))
    sB_layout_staged = cute.tile_to_shape(b_atom, (BLK_N, BLK_K, NUM_STAGES), order=(0, 1, 2))
    sD_layout = cute.tile_to_shape(d_atom, (BLK_M, BLK_N), order=(0, 1))

    sA_layout_one = cute.slice_(sA_layout_staged, (None, None, 0))
    sB_layout_one = cute.slice_(sB_layout_staged, (None, None, 0))

    # TMA atoms: B is multicast along cluster M (both members read the same B
    # tile; each CTA issues half of it and the DMA broadcasts to both smem
    # domains — lesson-12 recipe). A is plain G2S (CLUSTER_N == 1).
    a_g2s_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    b_g2s_op = (
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        if cutlass.const_expr(CLUSTER_M > 1)
        else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    )
    tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
        a_g2s_op,
        mA,
        sA_layout_one,
        (BLK_M, BLK_K),
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
        b_g2s_op,
        mB,
        sB_layout_one,
        (BLK_N, BLK_K),
        num_multicast=max(1, CLUSTER_M),
    )
    tma_atom_d, tma_tensor_d = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        mC,
        sD_layout,
        (BLK_M, BLK_N),
    )

    universal = cute.nvgpu.CopyUniversalOp()
    r2s_atom_d = cute.make_copy_atom(universal, out_dtype)
    r2s_tiled_copy_d = cute.make_tiled_copy_C(r2s_atom_d, tm)

    a_bytes = mA.element_type.width // 8
    b_bytes = mB.element_type.width // 8
    tx_count_ab = BLK_M * BLK_K * a_bytes + BLK_N * BLK_K * b_bytes

    @cute.struct
    class SharedStorage:
        mainloop_mbar_array: cute.struct.MemRange[cutlass.Int64, 2 * NUM_STAGES]
        sA: cute.struct.Align[
            cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout_staged)],
            1024,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout_staged)],
            1024,
        ]
        sD: cute.struct.Align[
            cute.struct.MemRange[out_dtype, cute.cosize(sD_layout)],
            1024,
        ]

    cta_layout_mnk = cute.make_layout((CLUSTER_M, CLUSTER_N, 1))
    cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))

    grid_n = (N + BLK_N - 1) // BLK_N
    grid_m = (M + BLK_M - 1) // BLK_M

    gemm_kernel(
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_d,
        tma_tensor_d,
        tm,
        r2s_tiled_copy_d,
        sA_layout_staged,
        sB_layout_staged,
        sD_layout,
        tx_count_ab,
        cta_layout_vmnk,
        cta_layout_mnk,
        grid_m,
        grid_n,
        acc_dtype,
        out_dtype,
        SharedStorage,
    ).launch(
        grid=(grid_n, grid_m, 1),
        block=(NUM_THREADS, 1, 1),
        cluster=(CLUSTER_N, CLUSTER_M, 1),
        stream=stream,
    )


__all__ = [
    "BLK_K",
    "BLK_M",
    "BLK_N",
    "CLUSTER_M",
    "CLUSTER_N",
    "NUM_STAGES",
    "NUM_THREADS",
    "gemm",
    "gemm_kernel",
]
