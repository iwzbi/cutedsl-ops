"""Block-tiled GEMM scaffold (CuTe DSL).

Implements ``C[m, n] = A[m, k] @ B^T`` where ``A`` is ``(M, K)`` row-major and
``B`` is stored **transposed** as ``(N, K)`` row-major (so the kernel reads
``B`` directly without a transpose pass). ``C`` is ``(M, N)`` row-major.

Architecture (Ampere sm_80 baseline; switch the atom to a warpgroup op for sm_90):

  - Three-level tiling. One thread block computes one ``(BLOCK_M, BLOCK_N)``
    output tile; the K dimension is walked in ``BLOCK_K`` steps through shared
    memory (gmem -> smem -> registers -> MMA -> ...).
  - TiledMma from a warp-level ``MmaF16BF16Op`` fp16 MMA atom.
  - TiledCopy for g2s (A/B) and the r2g epilogue (C); s2r uses the MMA's own
    fragments (``make_fragment_A/B``) or an explicit ``AutoVectorizingCopy``.

This file is a SCAFFOLD: the host ``@cute.jit`` entry builds the TiledMma and
launches the grid; the device kernel body is a guided TODO. Fill in the
``# TODO(practice)`` sections to obtain a correct kernel. Until you do, the
harness in ``run_gemm.py`` will report ``Failed`` (the kernel writes nothing).

Verified API notes:
  * cute.local_tile(mA, tiler=(M,K), coord=(i,j)) keeps the static tile shape.
  * cute.make_tiled_mma(op, atom_layout_mnk=(T,T,T)) tiles the atom across warps.
  * cute.copy(tiled_copy, src, dst)  for g2s / s2r / r2g tile-level copies.
  * cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)  computes D <- A*B + C.
  * cute.arch.cp_async_commit_group()/cp_async_wait_group() for cp.async; or
    cute.arch.barrier() / cute.sync() for the smem fence between stages.
"""

from __future__ import annotations

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import cute


# ---------------------------------------------------------------------------
# Tile sizes — tune freely. These are plain module constants; M/N/K themselves
# are passed as Constexprs in the kernel signature.
# ---------------------------------------------------------------------------
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 16


@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tiled_copy_c: cute.TiledCopy,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
):
    """One block computes one (BLOCK_M, BLOCK_N) output tile; walk K in BLOCK_K steps."""
    tid, _, _ = cute.arch.thread_idx()
    bid_m, bid_n, _ = cute.arch.block_idx()

    # --- 1. Project the per-block tiles out of global memory -----------------
    # gA = cute.local_tile(mA, (BLOCK_M, BLOCK_K), (bid_m, 0))   # (BLOCK_M, K)
    # gB = cute.local_tile(mB, (BLOCK_N, BLOCK_K), (bid_n, 0))   # (BLOCK_N, K)
    # gC = cute.local_tile(mC, (BLOCK_M, BLOCK_N), (bid_m, bid_n))  # (BLOCK_M, BLOCK_N)
    # TODO(practice): replace the placeholders above with the real local_tile calls.
    #   The leading K axis stays dynamic (use a dynamic tile of size K) so the
    #   mainloop below can stride along it.
    tiler = (BLOCK_M, BLOCK_N, BLOCK_K)
    gA = cute.local_tile(mA, tiler=tiler, c)

    # --- 2. Allocate shared-memory staging for one A and one B stripe --------
    # Use cute.make_tensor over a shared-memory allocator with the layout
    # produced by tiled_copy_a/b's smem layout, e.g.:
    #   smem_a = cute.make_tensor(
    #       cute.make_smem_allocator(), tiled_copy_a.layout_smem_A)
    # TODO(practice): allocate A_smem (BLOCK_M, BLOCK_K) and B_smem (BLOCK_N, BLOCK_K).

    # --- 3. Partition the TiledMma + copies across this thread --------------
    # thr_mma = tiled_mma.get_slice(tid)
    # tCgA = thr_mma.partition_A(gA); tCgB = thr_mma.partition_B(gB); tCgC = thr_mma.partition_C(gC)
    # tCrA = tiled_mma.make_fragment_A(tCgA)
    # tCrB = tiled_mma.make_fragment_B(tCgB)
    # tCrC = tiled_mma.make_fragment_C(tCgC)
    # tCrC.fill(0.0)
    # thr_copy_a = tiled_copy_a.get_slice(tid); tCsA = thr_copy_a.partition_S(smem_a); ...
    # TODO(practice): partition MMA + the g2s / s2r copies; clear the accumulator.

    # --- 4. K mainloop: gmem -> smem -> registers -> MMA, per BLOCK_K stripe -
    # for k in range(0, K, BLOCK_K):
    #     cute.copy(tiled_copy_a, gA_k, smem_a)   # g2s
    #     cute.copy(tiled_copy_b, gB_k, smem_b)
    #     cute.arch.cp_async_commit_group(); cute.arch.cp_async_wait_group(0)
    #     cute.sync()                            # smem fence
    #     cute.copy(tiled_copy_s2r_a, smem_a, tCrA)   # s2r
    #     cute.copy(tiled_copy_s2r_b, smem_b, tCrB)
    #     cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)  # D <- A*B + C
    #     # advance gA/gB along K for the next stripe
    # TODO(practice): implement the K mainloop. (Single-buffer first; add a
    #   software-pipelined double buffer once it's correct.)

    # --- 5. Epilogue: write accumulator back to gmem -------------------------
    # cute.copy(tiled_copy_c, tCrC, tCgC)
    # TODO(practice): store the result (consider a vectorized r2g copy / TMA on sm_90).


@cute.jit
def gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: CUstream,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
):
    """Host entry: build the TiledMma + TiledCopys and launch the grid."""
    # SM80 warp-level mma.m16n8k8.f16.f16.f16 — fp16 in, fp16 accumulator.
    # Swap for cute.nvgpu.warpgroup.MmaF16BF16Op(...) on Hopper (sm_90).
    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float16, (16, 8, 8))
    tiled_mma = cute.make_tiled_mma(op, atom_layout_mnk=(2, 2, 1))
    num_threads = tiled_mma.size

    # TODO(practice): build the g2s / s2r / r2g TiledCopys with cute.make_tiled_copy
    #   and pass them to gemm_kernel. Pick a 128-bit vectorized g2s copy and an
    #   AutoVectorizingCopy (or ldmatrix) for s2r. Examples:
    #   tiled_copy_a = cute.make_tiled_copy(
    #       cute.nvgpu.warp.LdMatrix8x8x16bOp(...), copy_layout_A, layout_A_smem)
    # tiled_copy_a = ...   # g2s A
    # tiled_copy_b = ...   # g2s B
    # tiled_copy_c = ...   # r2g C
    tiled_copy_a = None
    tiled_copy_b = None
    tiled_copy_c = None

    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N, 1)
    gemm_kernel(
        mA,
        mB,
        mC,
        tiled_mma,
        tiled_copy_a,
        tiled_copy_b,
        tiled_copy_c,
        M,
        N,
        K,
    ).launch(
        grid=grid,
        block=(num_threads, 1, 1),
        stream=stream,
    )


__all__ = ["BLOCK_K", "BLOCK_M", "BLOCK_N", "gemm", "gemm_kernel"]
