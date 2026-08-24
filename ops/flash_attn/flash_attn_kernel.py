"""FlashAttention forward scaffold (CuTe DSL).

Implements the forward pass of FlashAttention (v2 online-softmax):

    O[bh, m, :] = softmax_mn( Q[bh,m,:] @ K[bh,:,:]^T / sqrt(d) )  @  V[bh,:,:]

with an optional causal mask. Tensors are 3-D ``(*, M, D)`` where the leading
axis folds ``batch * n_heads`` together (the harness reshapes 4-D ``(B,H,S,D)``
to ``(B*H, S, D)``). One thread block computes one Q-block ``Br`` rows of a
single ``(bh)`` row, scanning the KV dimension in ``Bc`` steps with the online
softmax rescale.

Target: Hopper sm_90 (warpgroup MMA + TMA). On Ampere the algorithm still
works with a warp-level atom but performance will be limited.

This file is a SCAFFOLD: the host ``@cute.jit`` entry builds the TiledMma and
launches the grid; the device kernel body is a guided TODO. The exact MMA
instruction shape and smem layouts are design choices for you to make.

Algorithm (per Q-block ``i`` over the inner ``j`` loop):

  load Qi -> registers/smem; init Oi=0, mi=-inf, li=0
  for j in 0..ceil(M/Bc):                  # causal: j <= i
      load Kj, Vj -> smem
      Sij = (Qi @ Kj^T) / sqrt(d)          # (Br, Bc) MMA, accumulator
      apply causal mask -> Sij (set masked to -inf)
      m_new = max(mi, rowmax(Sij))
      P   = exp(Sij - m_new)               # softmax numerator (Br, Bc)
      t   = rowsum(P)
      Oi  = Oi * exp(mi - m_new)           # rescale running output
      li  = li * exp(mi - m_new) + t
      Oi  = Oi + P @ Vj                    # (Br, D) MMA
      mi  = m_new
  Oi = Oi / li
  store Oi
"""

from __future__ import annotations

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import cute


# Q-block rows and KV-block cols; head dim is the contraction/inner dims.
BR = 64
BC = 64
D = 64


@cute.kernel
def flash_attn_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    tiled_mma: cute.TiledMma,
    is_causal: cutlass.Constexpr[bool],
    BH: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    Dd: cutlass.Constexpr[int],
):
    """One block computes one Q-block (BR rows) of a single (batch, head) row."""
    tid, _, _ = cute.arch.thread_idx()
    bid_bh, bid_q, _ = cute.arch.block_idx()

    # scale = 1.0 / sqrt(Dd)  — use cute.math.sqrt / a Constexpr expr; see step 4.
    # TODO(practice): derive the softmax scale from Dd.

    # --- 1. Project tiles out of gmem ---------------------------------------
    # gQ = cute.local_tile(mQ, (BR, Dd), (bid_q, 0))           # needs the bh slice too
    # gK = cute.local_tile(mK, (BC, Dd), (j_block, 0))
    # gV = cute.local_tile(mV, (BC, Dd), (j_block, 0))
    # gO = cute.local_tile(mO, (BR, Dd), (bid_q, 0))
    # TODO(practice): slice the (bh) row and project per-block tiles. For 3-D
    #   tensors you can local_tile with a (1, BR, D) tiler and coord (bh, q, 0).

    # --- 2. Allocate shared memory for Kj, Vj (and optionally Qi) ------------
    # smem_k = ...   # (BC, D)
    # smem_v = ...   # (BC, D)
    # TODO(practice): allocate K/V smem staging (TMA descriptors on sm_90).

    # --- 3. Init running accumulators (m=rowmax, l=rowsum, O=output) -------
    # m_i = -inf; l_i = 0; acc_o = 0   (per-row, partitioned across the warps)
    # TODO(practice): init mi/l_i/Oi register tensors partitioned via thr_mma.

    # --- 4. Online-softmax mainloop over KV blocks j -----------------------
    # for j in range(0, M, BC):
    #     if is_causal and j > bid_q*BR: break          # causal early-exit
    #     load Kj, Vj -> smem; fence
    #     S = (Qi @ Kj^T) * scale                         # cute.gemm -> acc_s
    #     if is_causal: mask rows where m < j..j+BC      # set future positions -inf
    #     m_new = max(m_i, rowmax(S)); m_i = m_new
    #     P = exp(S - m_new)                             # cute.math elementwise
    #     t  = rowsum(P)
    #     acc_o *= exp(m_i_prev - m_new)                 # rescale (elementwise)
    #     l_i  = l_i * exp(m_i_prev - m_new) + t
    #     acc_o += P @ Vj                                 # cute.gemm -> acc_o
    # TODO(practice): implement the loop. Two MMAs (S=QK^T, O=PV), elementwise
    #   math via cute.math, row reductions via cute.reduce / warp shuffles.

    # --- 5. Normalize + epilogue -------------------------------------------
    # acc_o = acc_o / l_i
    # store acc_o -> gO
    # TODO(practice): divide by l_i and write the Q-block output back to gmem.


@cute.jit
def flash_attn(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    stream: CUstream,
    is_causal: cutlass.Constexpr[bool],
    BH: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    Dd: cutlass.Constexpr[int],
):
    """Host entry: build the (warpgroup) TiledMma and launch the grid."""
    # Hopper warpgroup MMA. On Ampere swap for cute.nvgpu.warp.MmaF16BF16Op.
    op = cute.nvgpu.warpgroup.MmaF16BF16Op(cutlass.Float16, cutlass.Float16, (128, 16, 16))
    tiled_mma = cute.make_tiled_mma(op, atom_layout_mnk=(1, 1, 1))
    num_threads = tiled_mma.size

    grid = (BH, (M + BR - 1) // BR, 1)
    flash_attn_kernel(
        mQ,
        mK,
        mV,
        mO,
        tiled_mma,
        is_causal,
        BH,
        M,
        Dd,
    ).launch(
        grid=grid,
        block=(num_threads, 1, 1),
        stream=stream,
    )


__all__ = ["BC", "BR", "D", "flash_attn", "flash_attn_kernel"]
