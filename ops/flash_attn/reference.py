"""Torch reference implementations for FlashAttention exercises.

Two tiers:
- **BF16**: uses ``torch.nn.functional.scaled_dot_product_attention`` (fp32
  internally) — the "true" answer.  Tolerance: ``atol=0.016``.
- **FP8**: mirrors the kernel's numerics — online softmax in fp32, then
  Px256 -> fp8 quant -> V matmul.  This deliberately reproduces the fp8
  rounding so the tolerance accounts for quantization, not just algorithmic
  error.  Tolerance: ``atol=0.05`` (prefill), ``0.1`` (decode).

Shapes:
- **Prefill**: ``(B, H, M, D)`` standard attention; ``M`` can be large.
- **Decode**: ``(B, H, M, D)`` with **paged KV** — ``M`` is small (1-4).
  KV pages live in ``(num_pages, H_kv, page_size, D)``; ``block_table``
  ``(B, max_blocks)`` int32 indexes into the pool (``-1`` = unused).

GQA is handled by repeating ``H_kv`` heads to ``H`` via ``repeat_interleave``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# BF16 references (torch SDPA — the "true" answer)
# ---------------------------------------------------------------------------


def ref_prefill_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Prefill reference: ``(B, H, M, D) x (B, H_kv, N, D) -> (B, H, M, D)`` fp32.

    Uses torch SDPA (fp32 internally).  Handles GQA via ``repeat_kv``.
    """
    H = q.shape[1]
    k = repeat_kv(k, H)
    v = repeat_kv(v, H)
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=is_causal, scale=scale)


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
# FP8 references (mirror kernel numerics)
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


def ref_prefill_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    q_scale: torch.Tensor | float = 1.0,
    k_scale: torch.Tensor | float = 1.0,
    v_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """FP8 prefill reference (mirrors kernel numerics).

    q: ``(B, H, M, D)`` fp8_e4m3, k/v: ``(B, H_kv, N, D)`` fp8_e4m3.
    Scales: ``q_scale`` per-token-per-head ``(B,H,M,1)`` or scalar;
    ``k_scale``, ``v_scale`` per-tensor scalar.
    Returns ``(B, H, M, D)`` fp32.
    """
    H = q.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    k = repeat_kv(k, H)
    v = repeat_kv(v, H)

    qf = q.float() * q_scale
    kf = k.float() * k_scale
    vf = v.float() * v_scale

    S = (qf @ kf.transpose(-2, -1)) * scale

    if is_causal:
        M = qf.shape[-2]
        N = S.shape[-1]
        mask = torch.triu(torch.ones(M, N, device=q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))

    return _online_softmax_fp8(S, vf)


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
    "allclose",
    "gather_paged_kv",
    "lse_combine",
    "ref_decode_bf16",
    "ref_decode_fp8",
    "ref_prefill_bf16",
    "ref_prefill_fp8",
    "repeat_kv",
]
