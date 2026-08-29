"""Run + validate the block-tiled GEMM against torch.

Usage::

    python ops/gemm/run_gemm.py            # default 1024x1024x1024 fp16
    python ops/gemm/run_gemm.py 4096 4096 4096
    python ops/gemm/run_gemm.py --cuda-graphs 4096 4096 4096

The kernel in ``gemm_kernel.py`` is a scaffold until you implement its body;
until then this harness will print ``Failed`` (the kernel writes nothing).
"""

from __future__ import annotations

import sys

import cuda.bindings.driver as cuda_driver
import torch
from cutlass import cute
from cutlass.testing import JitArguments, benchmark, get_workspace_count


sys.path.insert(0, ".")
from common.bench import PRINT_LENGTH, compare_tensor, cuda_bench
from common.cute_runtime import make_cute_tensor, make_stream
from ops.gemm.gemm_kernel import gemm


def torch_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B^T, with B stored as (N, K) row-major."""
    return a.float() @ b.float().t()


def _cute_bench(
    compiled,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    *,
    use_cuda_graphs: bool = False,
) -> float:
    """Benchmark via cutlass.testing.benchmark with L2-cache flushing.

    Returns ms/call. Falls back to cuda_bench if the CuTe DSL benchmark
    encounters a stream-compatibility error (can happen when the compiled
    function's internal stream doesn't match the benchmark's expectation).
    """
    try:
        elt_size = a.element_size()
        workspace_bytes = (M * K + N * K + M * N) * elt_size
        ws_count = min(
            get_workspace_count(workspace_bytes, warmup_iterations=10, iterations=100),
            10,
        )

        def workspace_gen() -> JitArguments:
            wa = torch.randn(M, K, device="cuda", dtype=dtype) * 0.5
            wb = torch.randn(N, K, device="cuda", dtype=dtype) * 0.5
            wc = torch.zeros(M, N, device="cuda", dtype=dtype)
            return JitArguments(wa, wb, wc)

        torch_stream = torch.cuda.current_stream()
        cu_stream = cuda_driver.CUstream(torch_stream.cuda_stream)

        time_us = benchmark(
            compiled,
            warmup_iterations=10,
            iterations=100,
            stream=cu_stream,
            workspace_generator=workspace_gen,
            workspace_count=ws_count,
            use_cuda_graphs=use_cuda_graphs,
        )
        return time_us / 1e3  # us -> ms

    except (ValueError, RuntimeError, TypeError) as exc:
        print(f"  [benchmark] CuTe DSL benchmark failed ({exc}), falling back to cuda_bench".center(PRINT_LENGTH))
        return cuda_bench(compiled, a, b, c)


def run_case(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.float16,
    bench: bool = False,
    use_cuda_graphs: bool = False,
) -> bool:
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
        options="--enable-tvm-ffi --generate-line-info",
    )
    compiled(a, b, c)
    torch.cuda.synchronize()

    ref = torch_ref(a, b).to(dtype)
    ok = compare_tensor(c, ref, name=f"gemm {M}x{N}x{K}")

    if bench and ok:
        ms = _cute_bench(compiled, a, b, c, M, N, K, dtype, use_cuda_graphs=use_cuda_graphs)
        flops = 2.0 * M * N * K
        print(f" [gemm {M}x{N}x{K}] {ms:.4f} ms/call, {flops / ms / 1e9:,.1f} TFLOPS".center(PRINT_LENGTH, "-"))
    return ok


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU (sm_80+).")

    use_cuda_graphs = "--cuda-graphs" in sys.argv
    sys.argv = [x for x in sys.argv if x != "--cuda-graphs"]

    args = [int(x) for x in sys.argv[1:4]]
    shapes = [tuple(args)] if len(args) == 3 else [(1024, 1024, 1024)]

    counters = {"succeed": 0, "failed": 0}
    for M, N, K in shapes:
        if run_case(M, N, K, bench=True, use_cuda_graphs=use_cuda_graphs):
            counters["succeed"] += 1
        else:
            counters["failed"] += 1

    print(f"\n Summary: {counters['succeed']} succeed, {counters['failed']} failed ".center(PRINT_LENGTH, "="))
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
