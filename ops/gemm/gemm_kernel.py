"""Block-tiled GEMM scaffold — Hopper (sm_90) warpgroup-MMA + TMA edition.

Implements ``C[m, n] = A[m, k] @ B^T`` where ``A`` is ``(M, K)`` row-major and
``B`` is stored **transposed** as ``(N, K)`` row-major (so the kernel reads
``B`` directly without a transpose pass). ``C`` is ``(M, N)`` row-major.

Architecture (Hopper sm_90; the direct port of the Ampere baseline):

  - **WGMMA**: a warpgroup-level ``MmaF16BF16Op`` (64x256x16) tiled (2,1,1)
    over M -> one 128x256x16 CTA tile per issue. A and B are consumed from
    **shared memory as descriptors** (``OperandSource.SMEM``), so the load path
    must land A/B in smem with the swizzled layout the hardware expects.
  - **TMA load/store**: ``CopyBulkTensorTileG2SOp`` (gmem -> smem, issued by a
    single thread) for A/B; ``CopyBulkTensorTileS2GOp`` for the C epilogue.
  - **Pipeline**: ``cutlass.pipeline.PipelineTmaAsync`` (mbarrier-based) with
    NUM_STAGES buffers; producer = 1 thread issuing TMA, consumer = 8 warps.
  - **Accumulator**: lives in registers across all K-tiles; epilogue narrows it
    to the output dtype, R2S's it into smem, then warp 0 TMA-stores it out.

This file is a SCAFFOLD: the host ``@cute.jit`` entry is complete and
launches the grid; the device kernel body is a guided TODO. Fill in the
``# TODO(practice)`` sections to obtain a correct kernel. Until you do, the
harness in ``run_gemm.py`` will report ``Failed`` (the kernel writes nothing).

Reference implementations to mirror (read *before* writing the TODOs):
  * ../cutlass-notes/11-tma-load-store/cutedsl_tma_load_store.py  — TMA basics
  * ../cutlass-notes/13-warpgroup-mma/cutedsl_warpgroup_mma.py   — the full
    wgmma + TMA + pipeline mainloop this scaffold is a stripped port of.

Verified API notes:
  * WGMMA atom wants ``cute.make_tiled_mma(cute.make_mma_atom(op), ...)`` —
    ``make_mma_atom`` is NOT optional for warpgroup ops.
  * Swizzled smem layouts come from ``cutlass.utils.hopper_helpers``; feeding a
    plain row-major layout to WGMMA is a correctness trap.
  * ``make_tiled_tma_atom`` returns ``(atom, tensor)`` — pass BOTH to the kernel
    (the tensor carries the box/swizzle descriptor; ``cute.copy(atom, ...)``
    needs the atom, ``local_tile`` needs the tensor).
  * TMA stores are issued by exactly ONE thread (``elect_one``); WGMMA
    synchronization is ``warpgroup.fence / commit_group / wait_group``, not the
    cp.async helpers.
  * grid is ``(N-blocks, M-blocks)`` — the first grid dim strides N.
"""

from __future__ import annotations

import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
from cuda.bindings.driver import CUstream
from cutlass import cute, pipeline
from cutlass.cute.nvgpu.warpgroup import OperandMajorMode, OperandSource
from cutlass.utils.layout import LayoutEnum


# ---------------------------------------------------------------------------
# Tile sizes — Hopper WGMMA atom is 64x256x16; tiled (2,1,1) over M.
# Tune freely; M/N/K themselves are passed as Constexprs in the signature.
# ---------------------------------------------------------------------------
BLK_M = 128
BLK_N = 256
BLK_K = 64  # wgmma K=16, 4 sub-blocks per tile; 3-stage pipeline fits smem
NUM_STAGES = 3  # 3 stages = best overlap; 4 crashes (smem > 228KB)

# WGMMA atom is 64x256x16 with atom layout (2, 1, 1) over M -> CTA tile
# 128x256x16 per issue. Two MMA warpgroups => 256 MMA threads.
ATOM_LAYOUT_MNK = (2, 1, 1)
NUM_MMA_WARPGROUPS = ATOM_LAYOUT_MNK[0] * ATOM_LAYOUT_MNK[1] * ATOM_LAYOUT_MNK[2]
NUM_DMA_WARPGROUPS = 1  # 1 producer warpgroup for TMA loads
NUM_WARPGROUPS = NUM_DMA_WARPGROUPS + NUM_MMA_WARPGROUPS  # 3
NUM_THREADS_PER_WARPGROUP = 128
NUM_THREADS = NUM_MMA_WARPGROUPS * NUM_THREADS_PER_WARPGROUP  # 256 (MMA threads)
NUM_WARPS = NUM_THREADS // 32  # 8
NUM_MMA_THREADS = NUM_MMA_WARPGROUPS * NUM_THREADS_PER_WARPGROUP  # 256
MMA_NAMED_BARRIER_ID = 1
LOAD_REGISTER_REQUIREMENT = 40
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
    acc_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    shared_storage_cls: cutlass.Constexpr,
):
    """Split-K GEMM: grid (grid_n, grid_m, split_k), each CTA handles a K slice.

    split_k=1: standard GEMM, TMA S2G epilogue (overwrite).
    split_k>1: each CTA computes a partial result, atomic_add to output.
    """
    tid, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # --- 0. Allocate the CTA's smem (barriers + staged A/B + C tile) ---------
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage_cls)

    sA_full = storage.sA.get_tensor(
        sA_layout_staged.outer,
        swizzle=sA_layout_staged.inner,
    )
    sB_full = storage.sB.get_tensor(
        sB_layout_staged.outer,
        swizzle=sB_layout_staged.inner,
    )
    sD = storage.sD.get_tensor(sD_layout.outer, swizzle=sD_layout.inner)

    # --- 1. TMA mainloop barriers --------------------------------------------
    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mainloop_mbar_array.data_ptr(),
        num_stages=NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            NUM_WARPS,
        ),
        tx_count=tx_count_ab,
        cta_layout_vmnk=cta_layout_vmnk,
    )

    warpgroup_idx = cute.arch.make_warp_uniform(tid // NUM_THREADS_PER_WARPGROUP)
    is_producer = warpgroup_idx < NUM_DMA_WARPGROUPS

    tiler = (BLK_M, BLK_N, BLK_K)
    m_tile = bidy + bidz * grid_m
    gA = cute.local_tile(mA, tiler=tiler, coord=(bidy, bidx, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=tiler, coord=(bidy, bidx, None), proj=(None, 1, 1))
    gD = cute.local_tile(mD, tiler=tiler, coord=(m_tile, bidx, 0), proj=(1, 1, None))

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
    k_start = bidz * k_per_split
    k_end = k_start + k_per_split
    k_end = min(k_end, total_k_tiles)
    num_k_tiles = k_end - k_start

    mainloop_prod_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        NUM_STAGES,
    )
    mainloop_cons_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        NUM_STAGES,
    )

    if is_producer:
        cute.arch.warpgroup_reg_dealloc(LOAD_REGISTER_REQUIREMENT)
        if warp_idx == 0:
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
        warpgroup_thread_layout = cute.make_layout(
            NUM_MMA_WARPGROUPS,
            stride=NUM_THREADS_PER_WARPGROUP,
        )
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
                cute.arch.cp_async_bulk_wait_group(0, read=False)
        mma_named_barrier.arrive_and_wait()


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
    """Host entry: build the WGMMA TiledMma, TMA atoms, pipeline, and launch.

    The host side is complete; compare each piece against the kernel TODOs.
    ``mC`` doubles as the output tensor (C = A @ B^T, no separate D).
    """
    acc_dtype = cutlass.Float32  # fp32 accumulator, fp16 output (see below)
    out_dtype = mC.element_type

    # ----- TiledMMA (Hopper WGMMA, A from SMEM, K-major A/B) -----
    op = cute.nvgpu.warpgroup.MmaF16BF16Op(
        mA.element_type,
        acc_dtype,
        (64, BLK_N, 16),
        OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    tm = cute.make_tiled_mma(cute.make_mma_atom(op), ATOM_LAYOUT_MNK)

    # ----- Swizzled smem layouts (=> what the kernel's s*_layout args are) --
    # get_smem_layout_atom picks the swizzle atom from (major-extent, dtype,
    # K). tile_to_shape grows it to (BLK_M, BLK_K[, NUM_STAGES]).
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
    sA_layout_staged = cute.tile_to_shape(
        a_atom,
        (BLK_M, BLK_K, NUM_STAGES),
        order=(0, 1, 2),
    )
    sB_layout_staged = cute.tile_to_shape(
        b_atom,
        (BLK_N, BLK_K, NUM_STAGES),
        order=(0, 1, 2),
    )
    sD_layout = cute.tile_to_shape(d_atom, (BLK_M, BLK_N), order=(0, 1))

    # Single-stage layouts are what the TMA box descriptors are built from.
    sA_layout_one = cute.slice_(sA_layout_staged, (None, None, 0))
    sB_layout_one = cute.slice_(sB_layout_staged, (None, None, 0))

    # ----- TMA atoms / tensors (g2s for A/B, s2g for the C epilogue) ---------
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

    # ----- R2S copy atom (universal, built off the TiledMMA) -----------------
    universal = cute.nvgpu.CopyUniversalOp()
    r2s_atom_d = cute.make_copy_atom(universal, out_dtype)
    r2s_tiled_copy_d = cute.make_tiled_copy_C(r2s_atom_d, tm)

    # ----- Transaction byte counts for the mbarrier --------------------------
    a_bytes = mA.element_type.width // 8
    b_bytes = mB.element_type.width // 8
    tx_count_ab = BLK_M * BLK_K * a_bytes + BLK_N * BLK_K * b_bytes

    # ----- Shared storage: mbarriers + staged A/B + C tile -------------------
    @cute.struct
    class SharedStorage:
        # 2 Int64 mbarriers per stage (full + empty) for the mainloop pipeline.
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
        acc_dtype,
        out_dtype,
        SharedStorage,
    ).launch(
        grid=(grid_n, grid_m, split_k),
        block=(NUM_WARPGROUPS * NUM_THREADS_PER_WARPGROUP, 1, 1),
        stream=stream,
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
