"""MegaMoE scaffold (CuTe DSL).

The heart of [MegaMoE](https://arxiv.org/abs/2310.07308) / megablock-style
mixture-of-experts is a **grouped GEMM**: one matmul that processes every
expert's tokens with that expert's weight, instead of E separate GEMMs with
per-expert kernel launches. This file scaffolds that grouped GEMM; the harness
``run_megamoe.py`` wires it into a full top-1 MoE FFN (route -> grouped GEMM ->
activation -> grouped GEMM -> combine) and validates against a torch reference.

Grouped GEMM shapes (per-expert tensors packed on the leading expert axis):

    A : (E, T, K_in)     # tokens, sorted+padded to T per expert
    B : (E, K_in, K_out) # expert weights
    C : (E, T, K_out)    # outputs   C[e] = A[e] @ B[e]

One thread block computes one ``(expert, M-tile)`` and walks ``K_in`` in
``BLOCK_K`` steps through shared memory — i.e. the GEMM from ``ops/gemm`` with
an extra leading expert dimension selected by ``block_idx``.

This file is a SCAFFOLD: the host builds the TiledMma + grid; the device kernel
body is a guided TODO. The routing / softmax / activation / combine all live
in the torch harness (they are easy to add to the kernel later as a fusion
exercise).
"""

from __future__ import annotations

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import cute


# Per-expert M-tile and K-tile. T (tokens per expert, padded) and the feature
# sizes are passed as Constexprs.
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16


@cute.kernel
def grouped_gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tiled_copy_c: cute.TiledCopy,
    E: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K_in: cutlass.Constexpr[int],
    K_out: cutlass.Constexpr[int],
):
    """One block computes one (expert, M-tile) output; walks K_in in BLOCK_K steps."""
    tid, _, _ = cute.arch.thread_idx()
    bid_e, bid_m, _ = cute.arch.block_idx()

    # --- 1. Project the (expert, M-tile) tiles out of gmem ------------------
    # Select expert e = bid_e and an M-tile (bid_m) from each of A/B/C:
    #   gA = cute.local_tile(mA, (1, BLOCK_M, BLOCK_K), (bid_e, bid_m, 0))  # (1, BM, BK)
    #   gB = cute.local_tile(mB, (1, BLOCK_K, BLOCK_N), (bid_e, 0, 0))     # (1, BK, BN)
    #   gC = cute.local_tile(mC, (1, BLOCK_M, BLOCK_N), (bid_e, bid_m, 0))  # (1, BM, BN)
    # (or slice mA[bid_e] -> 2-D then local_tile like ops/gemm.)
    # TODO(practice): project per-expert tiles.

    # --- 2. Allocate shared-memory staging for one A/B K-stripe ------------
    # smem_a = ...  # (BLOCK_M, BLOCK_K)
    # smem_b = ...  # (BLOCK_K, BLOCK_N)
    # TODO(practice): allocate A/B smem.

    # --- 3. Partition the TiledMma + copies across this thread ------------
    # thr_mma = tiled_mma.get_slice(tid)
    # tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(gA))
    # tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(gB))
    # tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(gC))
    # tCrC.fill(0.0)
    # TODO(practice): partition MMA + copies; clear accumulator.

    # --- 4. K mainloop: gmem -> smem -> registers -> MMA, per BLOCK_K stripe -
    # for k in range(0, K_in, BLOCK_K):
    #     cute.copy(tiled_copy_a, gA_k, smem_a)
    #     cute.copy(tiled_copy_b, gB_k, smem_b)
    #     cute.sync()
    #     cute.copy(s2r_a, smem_a, tCrA)
    #     cute.copy(s2r_b, smem_b, tCrB)
    #     cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
    # TODO(practice): implement the K mainloop (same structure as ops/gemm).

    # --- 5. Epilogue: rmem -> gmem -----------------------------------------
    # cute.copy(tiled_copy_c, tCrC, tCgC)
    # TODO(practice): store the per-expert output tile.


@cute.jit
def grouped_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: CUstream,
    E: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K_in: cutlass.Constexpr[int],
    K_out: cutlass.Constexpr[int],
):
    """Host entry: build the TiledMma + TiledCopys and launch the grid."""
    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float16, (16, 8, 8))
    tiled_mma = cute.make_tiled_mma(op, atom_layout_mnk=(2, 2, 1))
    num_threads = tiled_mma.size

    # TODO(practice): build the g2s / s2r / r2g TiledCopys via cute.make_tiled_copy.
    tiled_copy_a = None
    tiled_copy_b = None
    tiled_copy_c = None

    grid = (E, (T + BLOCK_M - 1) // BLOCK_M, 1)
    grouped_gemm_kernel(
        mA,
        mB,
        mC,
        tiled_mma,
        tiled_copy_a,
        tiled_copy_b,
        tiled_copy_c,
        E,
        T,
        K_in,
        K_out,
    ).launch(
        grid=grid,
        block=(num_threads, 1, 1),
        stream=stream,
    )


__all__ = ["BLOCK_K", "BLOCK_M", "BLOCK_N", "grouped_gemm", "grouped_gemm_kernel"]
