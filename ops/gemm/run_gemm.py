"""Run + validate the block-tiled GEMM against torch.

Usage::

    python ops/gemm/run_gemm.py                       # default 1024³ fp16
    python ops/gemm/run_gemm.py 4096 4096 4096
    python ops/gemm/run_gemm.py --cuda-graphs 4096 4096 4096
    python ops/gemm/run_gemm.py --ncu 4096 4096 4096  # with ncu profiling

The kernel in ``gemm_kernel.py`` is a scaffold until you implement its body;
until then this harness will print ``Failed`` (the kernel writes nothing).
"""

from __future__ import annotations

import os
import sys

import torch
from cutlass import cute
from cutlass.testing import JitArguments, benchmark, get_workspace_count


sys.path.insert(0, ".")
from common.bench import (
    PRINT_LENGTH,
    KernelMeta,
    compare_tensor,
    cuda_bench,
    get_gpu_info,
    parse_ncu_output,
    print_bench_report,
    run_ncu_gui,
    run_ncu_profile,
)
from common.cute_runtime import make_cute_tensor, make_stream
from ops.gemm import gemm_kernel as kern
from ops.gemm.gemm_kernel import gemm


def torch_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B^T, with B stored as (N, K) row-major."""
    return a.float() @ b.float().t()


def _make_kernel_meta(total_tiles: int = 0) -> KernelMeta:
    """Build KernelMeta from the kernel module's compile-time constants."""
    return KernelMeta(
        name="GEMM",
        tile_dims={
            "BLK_M": kern.BLK_M,
            "BLK_N": kern.BLK_N,
            "BLK_K": kern.BLK_K,
            "NUM_STAGES": kern.NUM_STAGES,
        },
        block_threads=kern.NUM_WARPGROUPS * kern.NUM_THREADS_PER_WARPGROUP,
        block_description=(
            f"{kern.NUM_WARPGROUPS} warpgroups: {kern.NUM_DMA_WARPGROUPS} DMA + {kern.NUM_MMA_WARPGROUPS} MMA"
        ),
        grid_mode="persistent",
        extra={
            "MMA atom": f"wgmma m64n{kern.BLK_N}k16, layout={kern.ATOM_LAYOUT_MNK}",
            "total_tiles": str(total_tiles) if total_tiles else "",
        },
    )


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
    use_cupti: bool = False,
) -> tuple[float, int]:
    """Benchmark via cutlass.testing.benchmark with L2-cache flushing.

    Returns (ms/call, workspace_count). Falls back to cuda_bench on error.
    """
    warmup = 10
    iters = 100
    try:
        import cuda.bindings.driver as cuda_driver

        elt_size = a.element_size()
        workspace_bytes = (M * K + N * K + M * N) * elt_size
        ws_count = min(
            get_workspace_count(workspace_bytes, warmup_iterations=warmup, iterations=iters),
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
            warmup_iterations=warmup,
            iterations=iters,
            stream=cu_stream,
            workspace_generator=workspace_gen,
            workspace_count=ws_count,
            use_cuda_graphs=use_cuda_graphs,
            use_cupti=use_cupti,
        )
        return time_us / 1e3, ws_count  # us -> ms

    except (ValueError, RuntimeError, TypeError) as exc:
        print(f"  [benchmark] CuTe DSL benchmark failed ({exc}), falling back to cuda_bench".center(PRINT_LENGTH))
        ms = cuda_bench(compiled, a, b, c)
        return ms, 1


def _compile_and_run(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> object:
    """Compile the kernel and run it once. Returns the compiled callable."""
    print(f"Compiling CuTe DSL gemm({M}x{N}x{K}) ...")
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
    return compiled


def run_case(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.float16,
    bench: bool = False,
    use_cuda_graphs: bool = False,
    use_cupti: bool = False,
    use_ncu: bool = False,
    ncu_raw: bool = False,
    ncu_gui: bool = False,
) -> bool:
    torch.cuda.manual_seed_all(9527)
    a = torch.randn(M, K, device="cuda", dtype=dtype) * 0.5
    b = torch.randn(N, K, device="cuda", dtype=dtype) * 0.5
    c = torch.zeros(M, N, device="cuda", dtype=dtype)

    # When ncu is profiling this process, just compile + run once and exit.
    # This avoids mixing benchmark output with ncu's profiling report.
    if os.environ.get("NCU_PROFILING") == "1":
        _compile_and_run(a, b, c, M, N, K)
        return True

    compiled = _compile_and_run(a, b, c, M, N, K)

    ref = torch_ref(a, b).to(dtype)
    ok = compare_tensor(c, ref, name=f"gemm {M}x{N}x{K}")

    if bench and ok:
        ms, ws_count = _cute_bench(
            compiled,
            a,
            b,
            c,
            M,
            N,
            K,
            dtype,
            use_cuda_graphs=use_cuda_graphs,
            use_cupti=use_cupti,
        )

        # ncu profiling (optional)
        ncu_data = None
        if use_ncu:
            program_cmd = [sys.executable, os.path.abspath(__file__), str(M), str(N), str(K)]
            if ncu_gui:
                run_ncu_gui("gemm_kernel", program_cmd, output_file="gemm_profile.ncu-rep")
            else:
                ncu_text = run_ncu_profile("gemm_kernel", program_cmd)
                if ncu_raw:
                    print("\n--- ncu raw output ---")
                    print(ncu_text)
                else:
                    ncu_data = parse_ncu_output(ncu_text)

        # Build report metadata
        blocks_m = (M + kern.BLK_M - 1) // kern.BLK_M
        blocks_n = (N + kern.BLK_N - 1) // kern.BLK_N
        total_tiles = blocks_m * blocks_n
        num_sms = get_gpu_info()["num_sms"]
        grid_blocks = min(total_tiles, num_sms)
        meta = _make_kernel_meta(total_tiles=total_tiles)
        flops = 2 * M * N * K
        elt_size = a.element_size()
        gmem_bytes = (M * K + N * K + M * N) * elt_size

        timing_mode = "CUPTI" if use_cupti else ("CUDA Graphs" if use_cuda_graphs else "CUDA Events")

        print_bench_report(
            ms=ms,
            problem_shape=(M, N, K),
            dtype=dtype,
            flops=flops,
            gmem_bytes=gmem_bytes,
            ws_count=ws_count,
            meta=meta,
            grid_blocks=grid_blocks,
            timing_mode=timing_mode,
            ncu_data=ncu_data,
        )

    return ok


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU (sm_80+).")

    use_cuda_graphs = "--cuda-graphs" in sys.argv
    use_cupti = "--cupti" in sys.argv
    use_ncu = "--ncu" in sys.argv or "--ncu-raw" in sys.argv or "--ncu-gui" in sys.argv
    ncu_raw = "--ncu-raw" in sys.argv
    ncu_gui = "--ncu-gui" in sys.argv
    sys.argv = [x for x in sys.argv if x not in ("--cuda-graphs", "--cupti", "--ncu", "--ncu-raw", "--ncu-gui")]

    args = [int(x) for x in sys.argv[1:4]]
    shapes = [tuple(args)] if len(args) == 3 else [(1024, 1024, 1024)]

    counters = {"succeed": 0, "failed": 0}
    for M, N, K in shapes:
        if run_case(
            M,
            N,
            K,
            bench=True,
            use_cuda_graphs=use_cuda_graphs,
            use_cupti=use_cupti,
            use_ncu=use_ncu,
            ncu_raw=ncu_raw,
            ncu_gui=ncu_gui,
        ):
            counters["succeed"] += 1
        else:
            counters["failed"] += 1

    print(f"\n Summary: {counters['succeed']} succeed, {counters['failed']} failed ".center(PRINT_LENGTH, "="))
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
