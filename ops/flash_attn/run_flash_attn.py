"""Run + validate FlashAttention forward against torch SDPA.

Usage::

    python ops/flash_attn/run_flash_attn.py

Compares ``flash_attn`` (causal) against
``torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`` on a
small ``(B, H, S, D)`` problem. The kernel in ``flash_attn_kernel.py`` is a
scaffold until you implement its body; until then this harness reports
``Failed``.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F
from cutlass import cute


sys.path.append(".")
from common.bench import compare_tensor, cuda_bench
from common.cute_runtime import make_cute_tensor, make_stream
from ops.flash_attn.flash_attn_kernel import flash_attn


def torch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # PyTorch applies the 1/sqrt(D) scale automatically in SDPA.
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)


def run_case(B: int, H: int, S: int, Dd: int, dtype: torch.dtype = torch.float16, bench: bool = False) -> bool:
    torch.cuda.manual_seed_all(9527)
    scale = 1.0 / math.sqrt(Dd)
    q = (torch.randn(B, H, S, Dd, device="cuda", dtype=dtype) * (scale**0.5)).to(dtype)
    k = (torch.randn(B, H, S, Dd, device="cuda", dtype=dtype) * (scale**0.5)).to(dtype)
    v = torch.randn(B, H, S, Dd, device="cuda", dtype=dtype) * 0.5
    o = torch.zeros(B, H, S, Dd, device="cuda", dtype=dtype)

    # The kernel treats the leading B*H rows independently: reshape to (B*H, S, D).
    BH = B * H
    q3 = q.view(BH, S, Dd)
    k3 = k.view(BH, S, Dd)
    v3 = v.view(BH, S, Dd)
    o3 = o.view(BH, S, Dd)

    print(f"Compiling CuTe DSL flash_attn(B={B},H={H},S={S},D={Dd}, {dtype}) ...")
    compiled = cute.compile(
        flash_attn,
        make_cute_tensor(q3, leading_dim=q3.dim() - 1),
        make_cute_tensor(k3, leading_dim=k3.dim() - 1),
        make_cute_tensor(v3, leading_dim=v3.dim() - 1),
        make_cute_tensor(o3, leading_dim=o3.dim() - 1),
        make_stream(),
        True,  # is_causal
        BH,
        S,
        Dd,
    )
    compiled(q3, k3, v3, o3)
    torch.cuda.synchronize()

    ref = torch_ref(q, k, v).to(dtype)
    ok = compare_tensor(o, ref, name=f"flash_attn B={B}H={H}S={S}D={Dd}")

    if bench and ok:
        ms = cuda_bench(compiled, q3, k3, v3, o3)
        flops = 4.0 * BH * S * S * Dd  # 2*(QK^T) + 2*(PV) per head
        print(f" [flash_attn {B}x{H}x{S}x{Dd}] {ms:.4f} ms/call, {flops / ms / 1e9:,.1f} TFLOPS".center(100, "-"))
    return ok


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU (sm_90 recommended).")
    shapes = [(2, 4, 512, 64), (1, 8, 1024, 64)]
    counters = {"succeed": 0, "failed": 0}
    for shape in shapes:
        if run_case(*shape, bench=True):
            counters["succeed"] += 1
        else:
            counters["failed"] += 1
    print(f"\n Summary: {counters['succeed']} succeed, {counters['failed']} failed ".center(100, "="))
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
