"""Run + validate a top-1 MoE FFN built on the grouped-GEMM kernel.

Usage::

    python ops/megamoe/run_megamoe.py

The harness does token routing + softmax + the activation/combine on the host
and uses ``grouped_gemm`` (the scaffold kernel) for the two FFN matmuls. The
result is compared against a plain per-expert torch reference. Until you
implement the ``grouped_gemm_kernel`` body this reports ``Failed``.

Grouped GEMM packing (per-expert tokens on the leading axis, zero-padded):

    A1 : (E, T_pad, d_in)      # routed+sorted tokens per expert
    W1 : (E, d_in,   d_hidden) # up-proj weights
    H1 : (E, T_pad,  d_hidden) # = A1 @ W1, then GELU
    W2 : (E, d_hidden, d_out)  # down-proj weights
    H2 : (E, T_pad,  d_out)    # = H1 @ W2
    out : (N, d_out)           # gathered back & weighted by the gate prob
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from cutlass import cute


sys.path.append(".")
from common.bench import compare_tensor, cuda_bench
from common.cute_runtime import make_cute_tensor, make_stream
from ops.megamoe.megamoe_kernel import BLOCK_M, grouped_gemm


def torch_moe_ref(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Plain per-expert top-1 MoE FFN reference (loop over experts)."""
    N = x.shape[0]
    E = w1.shape[0]
    d_out = w2.shape[2]

    gate = F.softmax(x.float() @ w_gate.float(), dim=-1)  # (N, E)
    expert = gate.argmax(dim=-1)  # (N,)
    prob = gate.gather(-1, expert.unsqueeze(-1)).squeeze(-1)  # (N,)

    out = torch.zeros(N, d_out, device=x.device, dtype=x.dtype)
    for e in range(E):
        mask = expert == e
        if not mask.any():
            continue
        xe = x[mask].float()  # (n_e, d_in)
        h1 = F.gelu(xe @ w1[e].float())  # (n_e, d_hidden)
        h2 = h1 @ w2[e].float()  # (n_e, d_out)
        out[mask] = (h2 * prob[mask].unsqueeze(-1)).to(out.dtype)
    return out


def route_top1(x: torch.Tensor, w_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (sort_perm, expert, prob, offsets) for top-1 routing."""
    gate = F.softmax(x.float() @ w_gate.float(), dim=-1)
    expert = gate.argmax(dim=-1)  # (N,)
    prob = gate.gather(-1, expert.unsqueeze(-1)).squeeze(-1)
    sort_perm = torch.argsort(expert, stable=True)  # tokens grouped by expert id
    counts = torch.bincount(expert, minlength=w_gate.shape[1])
    offsets = torch.cumsum(counts, dim=0) - counts
    return sort_perm, expert, prob, offsets


def grouped_gemm_call(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    stream,
) -> None:
    """Compile (cached by the DSL) and run grouped_gemm: C[e] = A[e] @ B[e]."""
    E, T, K_in = a.shape
    K_out = c.shape[2]
    compiled = cute.compile(
        grouped_gemm,
        make_cute_tensor(a, leading_dim=a.dim() - 1),
        make_cute_tensor(b, leading_dim=b.dim() - 1),
        make_cute_tensor(c, leading_dim=c.dim() - 1),
        stream,
        E,
        T,
        K_in,
        K_out,
    )
    compiled(a, b, c)


def run_case(
    N: int = 512,
    E: int = 4,
    d_in: int = 128,
    d_hidden: int = 256,
    d_out: int = 128,
    dtype: torch.dtype = torch.float16,
    bench: bool = False,
) -> bool:
    torch.cuda.manual_seed_all(9527)
    x = torch.randn(N, d_in, device="cuda", dtype=dtype) * 0.5
    w_gate = torch.randn(d_in, E, device="cuda", dtype=dtype) * 0.1
    w1 = torch.randn(E, d_in, d_hidden, device="cuda", dtype=dtype) * (d_in**-0.5)
    w2 = torch.randn(E, d_hidden, d_out, device="cuda", dtype=dtype) * (d_hidden**-0.5)

    ref = torch_moe_ref(x, w_gate, w1, w2)

    # --- route + pack tokens per expert -------------------------------------
    sort_perm, expert, prob, offsets = route_top1(x, w_gate)
    counts = torch.bincount(expert, minlength=E)
    T_pad = int(((counts.max().item() + BLOCK_M - 1) // BLOCK_M) * BLOCK_M)

    a1 = torch.zeros(E, T_pad, d_in, device="cuda", dtype=dtype)
    for e in range(E):
        n_e = int(counts[e].item())
        if n_e:
            idx = sort_perm[offsets[e] : offsets[e] + n_e]
            a1[e, :n_e] = x[idx]

    # --- grouped GEMM 1: up-proj -------------------------------------------
    h1 = torch.zeros(E, T_pad, d_hidden, device="cuda", dtype=dtype)
    stream = make_stream()
    print(f"Compiling CuTe DSL grouped_gemm up-proj (E={E},T_pad={T_pad}, {dtype}) ...")
    grouped_gemm_call(a1, w1, h1, stream)
    torch.cuda.synchronize()
    h1 = F.gelu(h1.float()).to(dtype)  # host-side activation (fuse into kernel later)

    # --- grouped GEMM 2: down-proj ----------------------------------------
    h2 = torch.zeros(E, T_pad, d_out, device="cuda", dtype=dtype)
    print(f"Compiling CuTe DSL grouped_gemm down-proj (E={E},T_pad={T_pad}, {dtype}) ...")
    grouped_gemm_call(h1, w2, h2, stream)
    torch.cuda.synchronize()

    # --- gather back + weight by gate prob ---------------------------------
    out = torch.zeros(N, d_out, device="cuda", dtype=dtype)
    for e in range(E):
        n_e = int(counts[e].item())
        if not n_e:
            continue
        idx = sort_perm[offsets[e] : offsets[e] + n_e]
        out[idx] = (h2[e, :n_e].float() * prob[idx].unsqueeze(-1)).to(out.dtype)

    ok = compare_tensor(out, ref, name=f"megamoe N={N}E={E}d={d_in}/{d_hidden}/{d_out}")

    if bench and ok:
        ms = cuda_bench(lambda a, b, c: grouped_gemm_call(a, b, c, make_stream()), a1, w1, h1)
        flops = 2.0 * E * T_pad * d_in * d_hidden
        print(f" [megamoe up-proj] {ms:.4f} ms/call, {flops / ms / 1e9:,.1f} TFLOPS".center(100, "-"))
    return ok


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU (sm_80+).")
    if run_case(bench=True):
        print("\n Summary: 1 succeed, 0 failed ".center(100, "="))
    else:
        print("\n Summary: 0 succeed, 1 failed ".center(100, "="))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
