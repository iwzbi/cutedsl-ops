"""Block-tiled GEMM — Hopper (sm_90) warpgroup-MMA + TMA edition.

Implements ``C[m, n] = A[m, k] @ B^T`` where ``A`` is ``(M, K)`` row-major and
``B`` is stored **transposed** as ``(N, K)`` row-major. ``C`` is ``(M, N)``.

Persistent + Split-K: each CTA strides through multiple tiles in a 3D
``(split_k, grid_m, grid_n)`` tile space. Combines tail-wave elimination
(persistent) with K-dimension parallelism (split-K).
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
BLK_N = 256
BLK_K = 64
NUM_STAGES = 3

ATOM_LAYOUT_MNK = (2, 1, 1)
NUM_MMA_WARPGROUPS = ATOM_LAYOUT_MNK[0] * ATOM_LAYOUT_MNK[1] * ATOM_LAYOUT_MNK[2]
NUM_DMA_WARPGROUPS = 1
NUM_WARPGROUPS = NUM_DMA_WARPGROUPS + NUM_MMA_WARPGROUPS
NUM_THREADS_PER_WARPGROUP = 128
NUM_THREADS = NUM_MMA_WARPGROUPS * NUM_THREADS_PER_WARPGROUP
NUM_WARPS = NUM_THREADS // 32
NUM_MMA_THREADS = NUM_MMA_WARPGROUPS * NUM_THREADS_PER_WARPGROUP
MMA_NAMED_BARRIER_ID = 1
LOAD_REGISTER_REQUIREMENT = 24
MMA_REGISTER_REQUIREMENT = 232


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
    split_k: cutlass.Constexpr[int],
    grid_m: cutlass.Constexpr[int],
    grid_n: cutlass.Constexpr[int],
    num_persistent: cutlass.Constexpr[int],
    acc_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    shared_storage_cls: cutlass.Constexpr,
):
    """Persistent + Split-K: CTA stride loop over 3D tile space."""
    tid, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage_cls)

    sA_full = storage.sA.get_tensor(sA_layout_staged.outer, swizzle=sA_layout_staged.inner)
    sB_full = storage.sB.get_tensor(sB_layout_staged.outer, swizzle=sB_layout_staged.inner)
    sD = storage.sD.get_tensor(sD_layout.outer, swizzle=sD_layout.inner)

    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mainloop_mbar_array.data_ptr(),
        num_stages=NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_WARPS),
        tx_count=tx_count_ab,
        cta_layout_vmnk=cta_layout_vmnk,
    )

    warpgroup_idx = cute.arch.make_warp_uniform(tid // NUM_THREADS_PER_WARPGROUP)
    is_producer = warpgroup_idx < NUM_DMA_WARPGROUPS

    total_tiles = split_k * grid_m * grid_n
    tiles_per_cta = (total_tiles + num_persistent - 1) // num_persistent

    mainloop_prod_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        NUM_STAGES,
    )
    mainloop_cons_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        NUM_STAGES,
    )

    tiler = (BLK_M, BLK_N, BLK_K)
    mn_tiles = grid_m * grid_n

    if is_producer:
        cute.arch.warpgroup_reg_dealloc(LOAD_REGISTER_REQUIREMENT)
        if warp_idx == 0:
            for tile_iter in cutlass.range(tiles_per_cta, unroll=1):
                tile_idx = tile_iter * num_persistent + bidx
                if tile_idx < total_tiles:
                    split_idx = tile_idx // mn_tiles
                    mn_idx = tile_idx % mn_tiles
                    bid_m = mn_idx // grid_n
                    bid_n = mn_idx % grid_n
                    m_tile = bid_m + split_idx * grid_m

                    gA = cute.local_tile(mA, tiler=tiler, coord=(bid_m, bid_n, None), proj=(1, None, 1))
                    gB = cute.local_tile(mB, tiler=tiler, coord=(bid_m, bid_n, None), proj=(None, 1, 1))
                    gD = cute.local_tile(mD, tiler=tiler, coord=(m_tile, bid_n, 0), proj=(1, 1, None))

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
                    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_b,
                        0,
                        cute.make_layout(1),
                        sB_for_tma,
                        gB_for_tma,
                    )

                    total_k_tiles = cute.size(gA, mode=[2])
                    k_per_split = (total_k_tiles + split_k - 1) // split_k
                    k_start = split_idx * k_per_split
                    k_end = k_start + k_per_split
                    k_end = min(k_end, total_k_tiles)
                    num_k_tiles = k_end - k_start

                    for idx in cutlass.range(num_k_tiles, unroll=1):
                        mainloop_pipeline.producer_acquire(mainloop_prod_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA[None, k_start + idx],
                            tAsA[None, mainloop_prod_state.index],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB[None, k_start + idx],
                            tBsB[None, mainloop_prod_state.index],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
                        )
                        mainloop_pipeline.producer_commit(mainloop_prod_state)
                        mainloop_prod_state.advance()
    else:
        cute.arch.warpgroup_reg_alloc(MMA_REGISTER_REQUIREMENT)

        mma_warpgroup_idx = warpgroup_idx - NUM_DMA_WARPGROUPS
        warpgroup_thread_layout = cute.make_layout(NUM_MMA_WARPGROUPS, stride=NUM_THREADS_PER_WARPGROUP)
        thr_mma = tiled_mma.get_slice(warpgroup_thread_layout(mma_warpgroup_idx))

        tCsA = thr_mma.partition_A(sA_full)
        tCsB = thr_mma.partition_B(sB_full)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        num_k_blocks = cute.size(tCrA, mode=[2])

        mma_named_barrier = pipeline.NamedBarrier(
            barrier_id=MMA_NAMED_BARRIER_ID,
            num_threads=NUM_MMA_THREADS,
        )

        for tile_iter in cutlass.range(tiles_per_cta, unroll=1):
            tile_idx = tile_iter * num_persistent + bidx
            if tile_idx < total_tiles:
                split_idx = tile_idx // mn_tiles
                mn_idx = tile_idx % mn_tiles
                bid_m = mn_idx // grid_n
                bid_n = mn_idx % grid_n
                m_tile = bid_m + split_idx * grid_m

                gA = cute.local_tile(mA, tiler=tiler, coord=(bid_m, bid_n, None), proj=(1, None, 1))
                gB = cute.local_tile(mB, tiler=tiler, coord=(bid_m, bid_n, None), proj=(None, 1, 1))
                gD = cute.local_tile(mD, tiler=tiler, coord=(m_tile, bid_n, 0), proj=(1, 1, None))

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

                total_k_tiles = cute.size(gA, mode=[2])
                k_per_split = (total_k_tiles + split_k - 1) // split_k
                k_start = split_idx * k_per_split
                k_end = k_start + k_per_split
                k_end = min(k_end, total_k_tiles)
                num_k_tiles = k_end - k_start

                accumulators = cute.make_rmem_tensor(thr_mma.partition_C(gD).shape, acc_dtype)
                accumulators.fill(0.0)

                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for k_tile_idx in cutlass.range(num_k_tiles, unroll=1):
                    mainloop_pipeline.consumer_wait(mainloop_cons_state)
                    cute.nvgpu.warpgroup.fence()
                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        coord = (None, None, k_block_idx, mainloop_cons_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[coord], tCrB[coord], accumulators)
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                    mainloop_pipeline.consumer_release(mainloop_cons_state)
                    mainloop_cons_state.advance()

                # Epilogue: overlap TMA store with next tile's mainloop.
                # Wait for PREVIOUS tile's TMA store (overlapped with mainloop).
                if warp_idx == 5:
                    cute.arch.cp_async_bulk_wait_group(1, read=False)
                mma_named_barrier.arrive_and_wait()

                consumer_tid = tid - NUM_DMA_WARPGROUPS * NUM_THREADS_PER_WARPGROUP
                tCrD = cute.make_fragment_like(accumulators, out_dtype)
                tCrD.store(accumulators.load().to(out_dtype))
                thr_r2s_d = r2s_tiled_copy_d.get_slice(consumer_tid)
                tDrD_r2s = thr_r2s_d.retile(tCrD)
                tDsD_r2s = thr_r2s_d.partition_D(sD)
                cute.copy(r2s_tiled_copy_d, tDrD_r2s, tDsD_r2s)
                cute.arch.fence_proxy("async.shared", space="cta")
                mma_named_barrier.arrive_and_wait()
                if warp_idx == 5:
                    with cute.arch.elect_one():
                        cute.copy(tma_atom_d, tDsD[None, 0], tDgD[None, 0])
                        cute.arch.cp_async_bulk_commit_group()

        # Final wait for the last tile's TMA store.
        if warp_idx == 5:
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

    op = cute.nvgpu.warpgroup.MmaF8Op(
        mA.element_type,
        mB.element_type,
        acc_dtype,
        (64, BLK_N, 32),
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

    tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mA,
        sA_layout_one,
        (BLK_M, BLK_K),
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mB,
        sB_layout_one,
        (BLK_N, BLK_K),
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

    cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))

    grid_n = (N + BLK_N - 1) // BLK_N
    grid_m = (M + BLK_M - 1) // BLK_M
    total_tiles = split_k * grid_m * grid_n
    num_sms = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        0,
    )[1]
    num_persistent = min(total_tiles, num_sms)

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
        split_k,
        grid_m,
        grid_n,
        num_persistent,
        acc_dtype,
        out_dtype,
        SharedStorage,
    ).launch(
        grid=(num_persistent, 1, 1),
        block=(NUM_WARPGROUPS * NUM_THREADS_PER_WARPGROUP, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


__all__ = [
    "BLK_K",
    "BLK_M",
    "BLK_N",
    "NUM_DMA_WARPGROUPS",
    "NUM_MMA_WARPGROUPS",
    "NUM_STAGES",
    "NUM_THREADS",
    "NUM_THREADS_PER_WARPGROUP",
    "NUM_WARPGROUPS",
    "gemm",
    "gemm_kernel",
]
