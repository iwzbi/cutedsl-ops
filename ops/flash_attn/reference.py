"""Torch reference implementations for FlashAttention.

Two tiers:
- **BF16**: uses ``torch.nn.functional.scaled_dot_product_attention`` (fp32
  internally) — the "true" answer.  Tolerance: ``atol=0.016``.
- **FP8**: mirrors the kernel's numerics — online softmax in fp32, then
  Px256 -> fp8 quant -> V matmul.  This deliberately reproduces the fp8
  rounding so the tolerance accounts for quantization, not just algorithmic
  error.  Tolerance: ``atol=0.1`` (decode).

Decode: ``(B, H, M, D)`` with **paged KV** — ``M`` is small (1-4).  KV pages
live in ``(num_pages, H_kv, page_size, D)``; ``block_table`` ``(B, max_blocks)``
int32 indexes into the pool (``-1`` = unused).

``pack_varlen`` is the shared helper for varlen prefill: it flattens natural
``(B, H, S, D)`` tensors into the kernel's padded layout (see its docstring).
GQA is handled by repeating ``H_kv`` heads to ``H`` via ``repeat_interleave``.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Shared varlen prefill shapes
# ---------------------------------------------------------------------------

# Single authority for every prefill harness (run_prefill.py, compare_hpcops.py):
# both correctness and benchmarking run the SAME list — no separate "bench-only"
# shapes.  Each entry is varlen-form ``(H_q, H_kv, D, [seqlens...])``; the batch
# dimension is implicit (B = len(seqlens)) and every sequence is always causal.
# Coverage: single / multi batch, equal / unequal / BLK_M-misaligned lengths,
# MHA (H_q == H_kv) / GQA (H_q > H_kv) / single head (H_q == 1), large seq,
# and serving-like many-batch shapes.
PREFILL_SHAPES = [
    # single batch
    (4, 4, 128, [512]),  # square-ish MHA
    (8, 8, 128, [1024]),  # bigger MHA
    (4, 1, 128, [512]),  # GQA 4:1
    (1, 1, 128, [512]),  # single head H_q == 1
    (8, 2, 128, [4096]),  # GQA 4:1, large seq
    (4, 4, 128, [4096]),  # longest seq
    # multi batch
    (4, 4, 128, [512, 768]),  # unequal lengths
    (4, 4, 128, [200, 328]),  # misaligned (neither is a 64-multiple)
    (4, 1, 128, [256, 384, 512]),  # GQA + 3 batches, misaligned
    (4, 1, 128, [512, 512, 512, 512]),  # GQA + 4 equal batches
    (4, 4, 128, [2048, 2048]),  # longer seq per batch
    # serving-like (many batches)
    (4, 4, 128, [512] * 8),  # 8 batches, small seq
    (8, 8, 128, [1024] * 8),  # 8 batches MHA
    (4, 4, 128, [512] * 16),  # 16 batches
    # --- Llama-3-8B-class (32 Q / 8 KV heads, GQA 4:1, d=128) -------------
    # Industry-standard bench axis (FA / FlashInfer / NVIDIA blogs): keep the
    # TOTAL token budget ~16k and sweep (batch, seqlen).  These grids are big
    # (>=8k CTAs) — hpc-ops dispatches them to its warp_spec kernel, so the
    # †-region caveat applies; included for honest representativeness.
    (32, 8, 128, [512] * 32),  # 16k tokens, 512 seqs
    (32, 8, 128, [1024] * 16),  # 16k tokens, 1k seqs
    (32, 8, 128, [2048] * 8),  # 16k tokens, 2k seqs
    (32, 8, 128, [4096] * 4),  # 16k tokens, 4k seqs
    (32, 8, 128, [8192] * 2),  # 16k tokens, 8k seqs
    (32, 8, 128, [16384]),  # single 16k seq (long-context edge)
    # --- realistic varlen distributions (FlashInfer-style) -----------------
    (
        32,
        8,
        128,
        [736, 1920, 512, 3392, 1152, 2688, 640, 4096, 1536, 2944, 832, 2176, 1024, 3712, 1280, 2560],
    ),  # ~U(512,4k)
    (32, 8, 128, [128, 256, 384, 512, 640, 768, 1024, 1536, 2048, 3072, 4096, 6144]),  # zipf-like skew
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pack_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlens, blk_m: int = 64):
    """Pack natural ``(B, H, S, D)`` prefill tensors into the varlen kernel layout.

    The varlen kernel requires every batch's flattened start index
    (``cu_seqlens[b]``) to be a multiple of BLK_M (64) — see the kernel
    docstring.  This helper flattens each batch's Q/K segment with
    zero-padding to the next 64-multiple, pre-transposes V to
    ``(B, H_kv, D, S)`` (the PV B-operand's K-major layout), and returns the
    padded ``cu_seqlens`` offsets.

    Args:
        q: ``(B, H_q, S, D)`` — only rows ``[0, seqlens[b])`` of each batch
           are packed; ``S >= max(seqlens)``.
        k: ``(B, H_kv, S, D)`` — same row convention.
        v: ``(B, H_kv, S, D)`` — same row convention.
        seqlens: per-batch sequence lengths (list / tensor / CUDA tensor).
        blk_m: batch-alignment block size (kernel BLK_M = 64).

    Returns ``(q_cat, k_cat, v_t, o_cat, seqlens_t, cu_seqlens)``:
        q_cat: ``(total_padded, H_q, D)`` — flattened Q, zero-padded
        k_cat: ``(total_padded, H_kv, D)`` — flattened K, zero-padded
        v_t: ``(B, H_kv, D, S_pad)`` — V transposed (K-major for the PV MMA).
           ``S_pad`` is ``max(seqlens)`` rounded up to a ``blk_m`` multiple so
           the last per-batch V tile read by TMA stays in-bounds.
        o_cat: ``(total_padded, H_q, D)`` — zero-filled output buffer
        seqlens_t: int32 CUDA tensor ``(B,)`` — real lengths
        cu_seqlens: int32 CUDA tensor ``(B + 1,)`` — **padded** (64-aligned)
           offsets, i.e. the values the kernel indexes with
    """
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    device = q.device

    # (B, S, H, D) sequence-major views; only [0:seqlens[b]] rows are packed.
    q_nat = q.permute(0, 2, 1, 3).contiguous()
    k_nat = k.permute(0, 2, 1, 3).contiguous()

    segs_q, segs_k = [], []
    pad_offsets = torch.zeros(B + 1, dtype=torch.int64)
    for b in range(B):
        n_valid = int(seqlens[b])
        segs_q.append(q_nat[b, :n_valid])
        segs_k.append(k_nat[b, :n_valid])
        n_pad = (n_valid + blk_m - 1) // blk_m * blk_m - n_valid
        if n_pad:
            segs_q.append(torch.zeros(n_pad, H_q, D, device=device, dtype=q.dtype))
            segs_k.append(torch.zeros(n_pad, H_kv, D, device=device, dtype=k.dtype))
        pad_offsets[b + 1] = pad_offsets[b] + n_valid + n_pad

    q_cat = torch.cat(segs_q, dim=0).contiguous()
    k_cat = torch.cat(segs_k, dim=0).contiguous()
    o_cat = torch.zeros(int(pad_offsets[-1]), H_q, D, device=device, dtype=q.dtype)

    # (B, S, H_kv, D) sequence-major view, transposed to the PV K-major layout.
    # S_pad: TMA reads whole BLK_N-wide V tiles; a non-64-multiple max(seqlens)
    # would read past the per-batch S extent.  Pad S up to a blk_m multiple
    # (the extra columns are zero and masked out by the causal mask anyway).
    v_nat = v.permute(0, 2, 1, 3).contiguous()
    s_pad = (S + blk_m - 1) // blk_m * blk_m
    v_t = torch.zeros(B, H_kv, D, s_pad, device=device, dtype=v.dtype)
    for b in range(B):
        n_valid = int(seqlens[b])
        v_t[b, :, :, :n_valid] = v_nat[b, :n_valid, :, :].permute(1, 2, 0)

    seqlens_t = torch.tensor([int(s) for s in seqlens], device=device, dtype=torch.int32)
    cu_seqlens = pad_offsets.to(device).to(torch.int32)
    return q_cat, k_cat, v_t, o_cat, seqlens_t, cu_seqlens


def pick_split(ctas: int) -> int:
    """Split-K factor for the varlen prefill kernel, by grid shape.

    v4/v5/v6 A/B: split_k=2 wins on small grids (fewer than ~2 waves of 78
    SMs: up to 96 CTAs here) where per-CTA fixed cost dominates and extra CTAs
    fill the GPU; it loses on >=128-CTA grids where the GPU is already busy
    and the extra Q loads / partial traffic are pure cost.  See PERFLOG
    Step 4/5/6.
    """
    return 2 if ctas <= 96 else 1


def repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Repeat KV for GQA: ``(B, H_kv, N, D) -> (B, H, N, D)``."""
    if num_heads == x.shape[1]:
        return x
    rep = num_heads // x.shape[1]
    return x.repeat_interleave(rep, dim=1)


def gather_paged_kv(
    pages: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV into a dense tensor + validity mask.

    Args:
        pages: ``(num_pages, H_kv, page_size, D)``
        block_table: ``(B, max_blocks)`` int32, ``-1`` = unused

    Returns:
        dense: ``(B, H_kv, max_blocks * page_size, D)``
        valid_mask: ``(B, 1, max_blocks * page_size, 1)`` bool
    """
    B, max_blocks = block_table.shape
    H_kv = pages.shape[1]
    page_size = pages.shape[2]

    ids = block_table.clamp(min=0).long()  # (B, max_blocks)
    gathered = pages[ids]  # (B, max_blocks, H_kv, page_size, D)
    dense = gathered.permute(0, 2, 1, 3, 4).reshape(B, H_kv, max_blocks * page_size, pages.shape[3])

    valid = block_table >= 0  # (B, max_blocks)
    valid_mask = valid.unsqueeze(1).repeat_interleave(page_size, dim=-1)
    valid_mask = valid_mask.unsqueeze(-1)  # (B, 1, N, 1)
    return dense, valid_mask


# ---------------------------------------------------------------------------
# Decode reference (paged KV)
# ---------------------------------------------------------------------------


def ref_decode_bf16(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    block_table: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode reference with paged KV.

    q: ``(B, H, M, D)`` — ``M`` is small (1-4).
    k_pages, v_pages: ``(num_pages, H_kv, page_size, D)``
    block_table: ``(B, max_blocks)`` int32, ``-1`` = unused.

    Returns ``(B, H, M, D)`` fp32.
    """
    H = q.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    k_dense, valid_mask = gather_paged_kv(k_pages, block_table)
    v_dense, _ = gather_paged_kv(v_pages, block_table)

    k_dense = repeat_kv(k_dense, H)
    v_dense = repeat_kv(v_dense, H)
    valid_mask = valid_mask.expand(-1, H, -1, -1)  # (B, H, N, 1)

    qf = q.float()
    kf = k_dense.float()
    vf = v_dense.float()

    S = (qf @ kf.transpose(-2, -1)) * scale  # (B, H, M, N)
    # Mask invalid KV positions (from -1 block_table entries)
    S = S.masked_fill(~valid_mask.transpose(-2, -1), float("-inf"))
    if is_causal:
        M = qf.shape[-2]
        N = S.shape[-1]
        causal = torch.triu(torch.ones(M, N, device=q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(causal, float("-inf"))

    # Softmax
    S_max = S.max(dim=-1, keepdim=True).values
    P = torch.exp(S - S_max)
    P = P * valid_mask.squeeze(-1).unsqueeze(-2)
    l = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return (P @ vf) / l


# ---------------------------------------------------------------------------
# FP8 reference (mirror kernel numerics)
# ---------------------------------------------------------------------------


def _online_softmax_fp8(
    S: torch.Tensor,
    v: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Online softmax + Px256 fp8 quant + V matmul (mirrors kernel).

    Args:
        S: ``(B, H, M, N)`` fp32 — already scaled QK^T
        v: ``(B, H, N, D)`` fp32 — V (already cast from fp8 + scaled)
        valid_mask: ``(B, H, N, 1)`` bool — True where KV is valid (decode)

    Returns ``(B, H, M, D)`` fp32.
    """
    if valid_mask is not None:
        S = S.masked_fill(~valid_mask.transpose(-2, -1), float("-inf"))

    # Online softmax in fp32
    m = S.max(dim=-1, keepdim=True).values
    P = torch.exp(S - m)

    if valid_mask is not None:
        P = P * valid_mask.squeeze(-1).unsqueeze(-2)

    l = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # P pre-scale by 256, quantize to fp8_e4m3, back to fp32
    P_scaled = (P * 256.0).to(torch.float8_e4m3fn)
    P_quant = P_scaled.float() / 256.0

    # V matmul with quantized P (fp32 accumulate)
    return (P_quant @ v) / l


def ref_decode_fp8(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    block_table: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    q_scale: torch.Tensor | float = 1.0,
    k_scale: torch.Tensor | float = 1.0,
    v_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """FP8 decode reference with paged KV (mirrors kernel numerics).

    q: ``(B, H, M, D)`` fp8_e4m3 — ``M`` small (1-4).
    k_pages, v_pages: ``(num_pages, H_kv, page_size, D)`` fp8_e4m3.
    block_table: ``(B, max_blocks)`` int32.
    Returns ``(B, H, M, D)`` fp32.
    """
    H = q.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    k_dense, valid_mask = gather_paged_kv(k_pages, block_table)
    v_dense, _ = gather_paged_kv(v_pages, block_table)

    k_dense = repeat_kv(k_dense, H)
    v_dense = repeat_kv(v_dense, H)
    valid_mask = valid_mask.expand(-1, H, -1, -1)

    qf = q.float() * q_scale
    kf = k_dense.float() * k_scale
    vf = v_dense.float() * v_scale

    S = (qf @ kf.transpose(-2, -1)) * scale

    if is_causal:
        M = qf.shape[-2]
        N = S.shape[-1]
        mask = torch.triu(torch.ones(M, N, device=q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))

    return _online_softmax_fp8(S, vf, valid_mask)


# ---------------------------------------------------------------------------
# Comparison helper (mirrors hpc-ops utils.allclose)
# ---------------------------------------------------------------------------


def allclose(
    ref: torch.Tensor,
    real: torch.Tensor,
    *,
    atol: float,
    rtol: float = 1e-5,
    name: str = "op",
) -> bool:
    """Compare ref vs real in fp32.  On failure print top-10 worst positions.

    Returns ``True`` when ``max|ref - real| <= atol + rtol * |ref|``.
    """
    ref = ref.float()
    real = real.float()
    diff = (ref - real).abs()
    max_diff = diff.max().item()
    tol_val = atol + rtol * ref.abs()
    ok = (diff <= tol_val).all().item()

    status = "Success" if ok else "Failed"
    print(f" [{name}] {status}, atol={atol}, max diff = {max_diff:.5f} ".center(80, "-"))

    if not ok:
        flat_diff = diff.flatten()
        flat_ref = ref.flatten()
        flat_real = real.flatten()
        topk_vals, topk_idx = flat_diff.topk(10)
        print("  Top-10 worst positions (idx, ref, real, diff):")
        for i in range(min(10, len(topk_vals))):
            idx = topk_idx[i].item()
            print(
                f"    [{idx}] ref={flat_ref[idx].item():.6f}, "
                f"real={flat_real[idx].item():.6f}, diff={topk_vals[i].item():.6f}"
            )
    return ok


# ---------------------------------------------------------------------------
# Split-K LSE combine (for decode exercises 3 & 5)
# ---------------------------------------------------------------------------


def lse_combine(
    o_partials: torch.Tensor,
    lse_partials: torch.Tensor,
) -> torch.Tensor:
    """LSE-weighted combine of split-K partials.

    Args:
        o_partials: ``(kSplitK, B, H, M, D)`` — partial outputs from each split
        lse_partials: ``(kSplitK, B, H, M)`` — partial log-sum-exp values

    Formula::

        LSE_global = log(sum(exp(LSE_s)))
        alpha_s = exp(LSE_s - LSE_global)
        O = sum(O_s * alpha_s)

    Returns ``(B, H, M, D)`` fp32.
    """
    lse = lse_partials.permute(1, 2, 3, 0)  # (B, H, M, kSplitK)
    lse_max = lse.max(dim=-1, keepdim=True).values
    alpha = torch.exp(lse - lse_max)
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
    alpha_sum = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    alpha = alpha / alpha_sum

    o = o_partials.permute(1, 2, 3, 4, 0).float()  # (B, H, M, D, kSplitK)
    return (o * alpha.unsqueeze(-2)).sum(dim=-1)


__all__ = [
    "PREFILL_SHAPES",
    "allclose",
    "gather_paged_kv",
    "lse_combine",
    "pack_varlen",
    "pick_split",
    "ref_decode_bf16",
    "ref_decode_fp8",
    "repeat_kv",
]
