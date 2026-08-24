"""Correctness + timing helpers shared across operators.

Every ``run_*.py`` harness reuses these so the output format stays consistent:
a centered ``Success``/``Failed`` line carrying ``Max diff``, ``Mean diff`` and
relative error (RE) against a torch reference.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


PRINT_LENGTH = 100


def relative_error(target: torch.Tensor, ref: torch.Tensor, *, eps: float = 1e-8) -> float:
    """Frobenius relative error: ||target - ref|| / ||ref||."""
    diff = target - ref
    norm_diff = torch.norm(diff, p=2)
    norm_diff_ref = torch.norm(ref, p=2)
    return (norm_diff / (norm_diff_ref + eps)).item()


def compare_tensor(
    kernel_output: torch.Tensor,
    ref_output: torch.Tensor,
    *,
    name: str = "op",
    tol: float = 1e-2,
) -> bool:
    """Compare a kernel output to a torch reference and print a verdict line.

    Returns ``True`` when ``re < tol``. On failure it also dumps the first few
    elements of each side for quick inspection.
    """
    kernel_output = kernel_output.float()
    ref_output = ref_output.float()
    max_diff = torch.max(torch.abs(ref_output - kernel_output))
    mean_diff = torch.mean(torch.abs(ref_output - kernel_output))
    re = relative_error(kernel_output, ref_output)
    ok = re < tol
    status = "Success" if ok else "Failed"
    if not ok:
        print(f" [{name}] Kernel: {tuple(kernel_output.shape)} ".center(PRINT_LENGTH, "-"))
        print(kernel_output.flatten()[:8])
        print(f" [{name}] Reference: {tuple(ref_output.shape)} ".center(PRINT_LENGTH, "-"))
        print(ref_output.flatten()[:8])
    print(
        f" [{name}] {status}, Max diff = {max_diff:.5f}, Mean diff = {mean_diff:.5f}, RE = {re * 100:.2f}% ".center(
            PRINT_LENGTH, "-"
        )
    )
    return ok


def cuda_bench(fn: Callable, *args, warmup: int = 5, iters: int = 100) -> float:
    """Time ``fn(*args)`` with CUDA events; return the median ms per call."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(torch.median(torch.tensor(times)).item())


__all__ = ["PRINT_LENGTH", "compare_tensor", "cuda_bench", "relative_error"]
