"""Run + validate the block-tiled GEMM against torch.

Usage::

    python ops/gemm/run_gemm.py            # default 512x512x512 fp16
    python ops/gemm/run_gemm.py 1024 1024 1024

The kernel in ``gemm_kernel.py`` is a scaffold until you implement its body;
until then this harness will print ``Failed`` (the kernel writes nothing).
"""

from __future__ import annotations

import sys

import torch
from cutlass import cute


sys.path.append(".")
from common.bench import compare_tensor, cuda_bench
from common.cute_runtime import make_cute_tensor, make_stream
from ops.gemm.gemm_kernel import gemm


def torch_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B^T, with B stored as (N, K) row-major."""
    return a.float() @ b.float().t()


def run_case(M: int, N: int, K: int, dtype: torch.dtype = torch.float16, bench: bool = False) -> bool:
    torch.cuda.manual_seed_all(9527)
    a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.5
    b = torch.randn(N, K, device="cuda", dtype=dtype) * 0.5
    c = torch.zeros(M, N, device="cuda", dtype=dtype)

    # Mark the contiguous (last) axis as the static-stride leading dim so the
    # DSL infers strides without baking the full static shape in.
    print(f"Compiling CuTe DSL gemm({M}x{N}x{K}, {dtype}) ...")
    compiled = cute.compile(
        gemm,
        make_cute_tensor(a, leading_dim=1),
        make_cute_tensor(b, leading_dim=1),
        make_cute_tensor(c, leading_dim=1),
        make_stream(),
        M,
        N,
        K,
    )
    compiled(a, b, c)
    torch.cuda.synchronize()

    ref = torch_ref(a, b).to(dtype)
    ok = compare_tensor(c, ref, name=f"gemm {M}x{N}x{K}")

    if bench and ok:
        ms = cuda_bench(compiled, a, b, c)
        flops = 2.0 * M * N * K
        print(f" [gemm {M}x{N}x{K}] {ms:.4f} ms/call, {flops / ms / 1e9:,.1f} TFLOPS".center(100, "-"))
    return ok


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU (sm_80+).")
    args = [int(x) for x in sys.argv[1:4]]
    shapes = [tuple(args)] if len(args) == 3 else [(512, 512, 512), (1024, 1024, 1024)]

    counters = {"succeed": 0, "failed": 0}
    for M, N, K in shapes:
        if run_case(M, N, K, bench=True):
            counters["succeed"] += 1
        else:
            counters["failed"] += 1

    print(f"\n Summary: {counters['succeed']} succeed, {counters['failed']} failed ".center(100, "="))
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
