"""Correctness, timing, and profiling helpers shared across operators.

This module provides a unified benchmark pipeline:

1. **Correctness** — ``compare_tensor`` prints a verdict against a torch reference.
2. **Timing** — ``cuda_bench`` or ``cutlass.testing.benchmark`` with L2 flushing.
3. **Theoretical analysis** — ``print_bench_report`` shows roofline, occupancy,
   bandwidth, grid/block structure from compile-time constants.
4. **ncu profiling** — ``run_ncu_profile`` runs Nsight Compute as a subprocess,
   ``parse_ncu_output`` extracts key metrics, ``print_ncu_report`` shows actual
   hardware data alongside the theoretical analysis and identifies bottlenecks.

Usage in ``run_*.py``::

    from common.bench import KernelMeta, print_bench_report, run_ncu_profile

    meta = KernelMeta(
        name="GEMM",
        tile_dims={"BLK_M": 128, "BLK_N": 256, "BLK_K": 64, "NUM_STAGES": 3},
        block_threads=384,
        block_description="3 warpgroups: 1 DMA + 2 MMA",
    )
    print_bench_report(ms, (M, N, K), dtype, flops, gmem_bytes, ws_count, meta, ncu_data=ncu_data)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

import cuda.bindings.driver as drv
import torch


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GPU hardware info (queried once, cached)
# ---------------------------------------------------------------------------

_GPU_INFO: dict | None = None

# Known peak FP16 TFLOPS (dense MMA) for common datacenter GPUs.
# Throttled variants (H20, A30) must be in this table — they share the same
# compute capability as their full counterparts but have fewer active TCs/SM,
# which no CUDA driver attribute exposes.
_PEAK_FP16_TFLOPS: dict[str, float] = {
    "NVIDIA H20": 148.0,
    "NVIDIA H100 80GB HBM3": 989.0,
    "NVIDIA H100": 989.0,
    "NVIDIA A100 80GB": 312.0,
    "NVIDIA A100": 312.0,
    "NVIDIA A100-SXM4-40GB": 312.0,
    "NVIDIA L40S": 362.0,
    "NVIDIA L40": 362.0,
}


def get_gpu_info() -> dict:
    """Query GPU hardware limits via CUDA driver ``cuDeviceGetAttribute``."""
    global _GPU_INFO  # noqa: PLW0603
    if _GPU_INFO is not None:
        return _GPU_INFO

    def attr(a) -> int:
        _, val = drv.cuDeviceGetAttribute(a, 0)
        return val

    props = torch.cuda.get_device_properties(0)
    name = props.name

    peak_fp16 = _PEAK_FP16_TFLOPS.get(name)
    if peak_fp16 is None:
        clock_mhz = attr(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE) / 1000
        num_sms = props.multi_processor_count
        # FMA/cycle/SM for the *full* (non-throttled) reference chip of each arch.
        #   sm_90 Hopper (H100):  1892 FMA/cycle/SM  → 989T @ 132SM 1.98GHz
        #   sm_80 Ampere (A100):  1024 FMA/cycle/SM  → 312T @ 108SM 1.41GHz
        if props.major == 9:
            fma_per_sm_per_cycle = 1892
        elif props.major == 8:
            fma_per_sm_per_cycle = 1024
        else:
            fma_per_sm_per_cycle = 0
        peak_fp16 = num_sms * fma_per_sm_per_cycle * 2 * clock_mhz * 1e-6

    mem_clock_mhz = attr(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) / 1000
    bus_width = attr(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    hbm_bw_gbs = 2 * mem_clock_mhz * 1e6 * bus_width / 8 / 1e9  # DDR

    _GPU_INFO = {
        "name": name,
        "compute_capability": f"sm_{props.major}{props.minor}",
        "num_sms": props.multi_processor_count,
        "clock_mhz": attr(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE) / 1000,
        "l2_cache_bytes": attr(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
        "max_smem_per_sm": attr(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
        "max_threads_per_sm": attr(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
        "max_blocks_per_sm": attr(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR),
        "max_regs_per_sm": attr(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR),
        "mem_clock_mhz": mem_clock_mhz,
        "bus_width_bits": bus_width,
        "hbm_bw_gbs": hbm_bw_gbs,
        "peak_fp16_tflops": peak_fp16,
    }
    return _GPU_INFO


# ---------------------------------------------------------------------------
# Kernel metadata
# ---------------------------------------------------------------------------


@dataclass
class KernelMeta:
    """Operator-specific metadata for the benchmark report.

    Each ``run_*.py`` fills this in with the kernel's compile-time constants.
    """

    name: str  # "GEMM", "FlashAttn", "MegaMoE"
    tile_dims: dict[str, int]  # {"BLK_M": 128, "BLK_N": 256, ...}
    block_threads: int  # threads per CTA
    block_description: str  # "3 warpgroups: 1 DMA + 2 MMA"
    extra: dict[str, str] = field(default_factory=dict)  # extra info lines


# ---------------------------------------------------------------------------
# Occupancy estimation
# ---------------------------------------------------------------------------


def theoretical_occupancy(
    smem_bytes: int,
    block_threads: int,
    gpu: dict,
) -> tuple[int, int, float, str]:
    """Compute theoretical occupancy from smem + thread + block HW limits.

    Registers-per-thread is unknown from the Python API (needs ncu).
    """
    blocks_smem = gpu["max_smem_per_sm"] // max(smem_bytes, 1)
    blocks_threads = gpu["max_threads_per_sm"] // block_threads
    blocks_hw = gpu["max_blocks_per_sm"]

    blocks = min(blocks_smem, blocks_threads, blocks_hw)
    active_threads = blocks * block_threads
    occ_pct = active_threads / gpu["max_threads_per_sm"] * 100

    limiters: list[str] = []
    vals = [("smem", blocks_smem), ("threads", blocks_threads), ("hw", blocks_hw)]
    mn = min(v for _, v in vals)
    for label, v in vals:
        if v == mn:
            if label == "smem":
                limiters.append(f"smem {smem_bytes // 1024}KB/{gpu['max_smem_per_sm'] // 1024}KB")
            elif label == "threads":
                limiters.append(f"threads {block_threads}/{gpu['max_threads_per_sm']}")
            else:
                limiters.append(f"hw {blocks_hw} blocks/SM")
    limiter = " + ".join(limiters) if limiters else "none"
    return blocks, active_threads, occ_pct, limiter


# ---------------------------------------------------------------------------
# Comprehensive benchmark report (theoretical analysis)
# ---------------------------------------------------------------------------


def print_bench_report(
    ms: float,
    problem_shape: tuple[int, ...],
    dtype: torch.dtype,
    flops: int,
    gmem_bytes: int,
    ws_count: int,
    meta: KernelMeta,
    *,
    grid_blocks: int,
    block_threads: int | None = None,
    timing_mode: str = "CUDA Events",
    warmup: int = 10,
    iterations: int = 100,
    ncu_data: dict | None = None,
) -> None:
    """Print a comprehensive performance + hardware + ncu report.

    Args:
        ms: measured milliseconds per call
        problem_shape: e.g. (M, N, K) for GEMM
        dtype: torch dtype
        flops: total FLOPs for the operation
        gmem_bytes: total global memory I/O in bytes
        ws_count: workspace count for L2 flushing
        meta: kernel metadata (tile dims, block info, etc.)
        grid_blocks: total number of CTAs in the grid
        block_threads: override meta.block_threads if needed
        timing_mode: "CUDA Events" / "CUDA Graphs" / "CUPTI"
        warmup, iterations: benchmark parameters
        ncu_data: parsed ncu output (from ``parse_ncu_output``), if available
    """
    gpu = get_gpu_info()
    bt = block_threads or meta.block_threads
    elt_size = torch.tensor([], dtype=dtype).element_size()
    dtype_str = str(dtype).removeprefix("torch.")

    shape_str = "x".join(str(s) for s in problem_shape)

    # --- compute metrics ---
    tflops = flops / ms / 1e9
    peak_fp16 = gpu["peak_fp16_tflops"]
    peak_pct = tflops / peak_fp16 * 100 if peak_fp16 else 0.0

    bandwidth_gbs = gmem_bytes / ms / 1e6
    hbm_pct = bandwidth_gbs / gpu["hbm_bw_gbs"] * 100 if gpu["hbm_bw_gbs"] else 0.0
    ai = flops / gmem_bytes if gmem_bytes else 0.0

    waves = grid_blocks / gpu["num_sms"]
    smem_bytes = _estimate_smem(meta, elt_size)
    smem_kb = smem_bytes / 1024
    occ_blocks, occ_threads, occ_pct, occ_limiter = theoretical_occupancy(smem_bytes, bt, gpu)

    l2_mb = gpu["l2_cache_bytes"] / (1024 * 1024)
    ws_bytes = gmem_bytes * ws_count

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  {meta.name} {shape_str} {dtype_str} | {gpu['name']} {gpu['compute_capability']} | {gpu['num_sms']} SMs")
    print(sep)

    # Performance
    print(f"  {'Time':<20} {ms:.4f} ms/call  ({timing_mode}, {warmup} warmup + {iterations} iters)")
    print(f"  {'TFLOPS':<20} {tflops:,.1f} / {peak_fp16:,.0f} peak ({peak_pct:.1f}%)")
    print()

    # Memory
    print(f"  {'Bandwidth':<20} {bandwidth_gbs:,.1f} GB/s ({hbm_pct:.1f}% of {gpu['hbm_bw_gbs']:,.0f} GB/s HBM)")
    print(f"  {'Arith intensity':<20} {ai:,.0f} FLOP/byte  ({'compute-bound' if ai > 10 else 'memory-bound'})")
    print(f"  {'gmem I/O':<20} {gmem_bytes / 1e6:,.1f} MB")
    print()

    # Grid / Block
    print(f"  {'Grid':<20} {grid_blocks} blocks → {waves:.1f} waves on {gpu['num_sms']} SMs")
    print(f"  {'Block':<20} {bt} threads ({meta.block_description})")
    tile_str = ", ".join(f"{k}={v}" for k, v in meta.tile_dims.items())
    print(f"  {'Tile':<20} {tile_str}")
    print(f"  {'SMEM/block':<20} {smem_kb:,.0f} KB / {gpu['max_smem_per_sm'] // 1024} KB per SM")
    print()

    # Occupancy (theoretical)
    print(
        f"  {'Occupancy (est)':<20} {occ_blocks} block/SM, {occ_threads}/{gpu['max_threads_per_sm']} threads ({occ_pct:.1f}%)"
    )
    print(f"  {'Limiter':<20} {occ_limiter}")
    if ncu_data and ncu_data.get("regs_per_thread") is not None:
        regs = int(ncu_data["regs_per_thread"])
        reg_usage = regs * bt / gpu["max_regs_per_sm"] * 100
        print(
            f"  {'Registers':<20} {regs}/thread x {bt} = {regs * bt:,}/{gpu['max_regs_per_sm']:,} ({reg_usage:.1f}%)"
        )
    else:
        print(f"  {'Registers':<20} unknown (use: --ncu)")
    print()

    # Extra operator-specific info
    if meta.extra:
        for label, val in meta.extra.items():
            print(f"  {label + ' (extra)':<20} {val}")
        print()

    # L2 / workspace
    print(f"  {'L2 cache':<20} {l2_mb:,.0f} MB")
    print(f"  {'Workspaces':<20} {ws_count} x {gmem_bytes / 1e6:,.1f} MB = {ws_bytes / 1e6:,.1f} MB (L2 flush)")
    print()

    # GPU hardware
    print(f"  {'GPU clock':<20} {gpu['clock_mhz']:,.0f} MHz")
    print(
        f"  {'HBM':<20} {gpu['mem_clock_mhz']:,.0f} MHz x {gpu['bus_width_bits']}-bit → {gpu['hbm_bw_gbs']:,.0f} GB/s"
    )
    print(
        f"  {'HW limits/SM':<20} smem {gpu['max_smem_per_sm'] // 1024}KB, threads {gpu['max_threads_per_sm']}, blocks {gpu['max_blocks_per_sm']}, regs {gpu['max_regs_per_sm']}"
    )

    # ncu actual data
    if ncu_data:
        print_ncu_report(ncu_data, gpu, peak_pct, occ_pct)

    print(f"{sep}")


def _estimate_smem(meta: KernelMeta, elt_size: int) -> int:
    """Estimate smem per CTA from kernel tile dims.

    Looks for BLK_M, BLK_N, BLK_K, NUM_STAGES in meta.tile_dims.
    Falls back to 0 if insufficient info.
    """
    td = meta.tile_dims
    if not all(k in td for k in ("BLK_M", "BLK_N", "BLK_K", "NUM_STAGES")):
        return 0
    align = 1024

    def align_up(x: int) -> int:
        return (x + align - 1) & ~(align - 1)

    offset = 0
    # mbar: 2 * NUM_STAGES * 8 bytes (Int64)
    offset = align_up(offset)
    offset += 2 * td["NUM_STAGES"] * 8
    # sA: BLK_M x BLK_K x NUM_STAGES x elt_size
    offset = align_up(offset)
    offset += td["BLK_M"] * td["BLK_K"] * td["NUM_STAGES"] * elt_size
    # sB: BLK_N x BLK_K x NUM_STAGES x elt_size
    offset = align_up(offset)
    offset += td["BLK_N"] * td["BLK_K"] * td["NUM_STAGES"] * elt_size
    # sD: BLK_M x BLK_N x elt_size
    offset = align_up(offset)
    offset += td["BLK_M"] * td["BLK_N"] * elt_size
    return offset


# ---------------------------------------------------------------------------
# ncu profiling (Nsight Compute subprocess + parser)
# ---------------------------------------------------------------------------

_NCU_PATHS = [
    os.environ.get("NCU_PATH", ""),
    "/opt/nvidia/nsight-compute/2026.2.1/ncu",
    "/usr/local/cuda/bin/ncu",
    "ncu",
]


def _find_ncu() -> str:
    """Find the ncu binary."""
    for p in _NCU_PATHS:
        if p and os.path.exists(p):
            return p
    # Last resort: check if ncu is on PATH
    found = shutil.which("ncu")
    if found:
        return found
    return ""


def run_ncu_profile(
    kernel_name: str,
    program_cmd: list[str],
    *,
    ncu_path: str | None = None,
    section_set: str = "full",
    launch_count: int = 1,
    launch_skip: int = 0,
    timeout: int = 600,
) -> str:
    """Run ncu as a subprocess and return its stdout text.

    Args:
        kernel_name: regex pattern to match the kernel name (e.g. "gemm_kernel")
        program_cmd: the program + args to profile (e.g. [".venv/bin/python", "ops/gemm/run_gemm.py", "4096", "4096", "4096"])
        ncu_path: path to ncu binary (auto-detected if None)
        section_set: "basic" or "full"
        launch_count: number of kernel launches to profile
        launch_skip: number of matching launches to skip before profiling
        timeout: subprocess timeout in seconds

    Returns:
        ncu's stdout text (the profiling report)
    """
    ncu = ncu_path or _find_ncu()
    if not ncu:
        raise FileNotFoundError("ncu not found. Set NCU_PATH env var or install Nsight Compute.")

    cmd = [
        ncu,
        "--set",
        section_set,
        "--target-processes",
        "all",
        "--kernel-name",
        f"regex:{kernel_name}",
        "--launch-skip",
        str(launch_skip),
        "--launch-count",
        str(launch_count),
        *program_cmd,
    ]

    # Signal to the target program that it's being profiled by ncu.
    # The program should check NCU_PROFILING env var and skip its own
    # benchmark/correctness output to avoid mixing with ncu's output.
    env = os.environ.copy()
    env["NCU_PROFILING"] = "1"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    return result.stdout


def parse_ncu_output(text: str) -> dict:
    """Parse key metrics from ncu text output.

    ncu prints metrics in a three-column table::

        Metric Name             Metric Unit Metric Value
        ----------------------- ----------- ------------
        Compute (SM) Throughput           %        90.93

    This parser uses regex to extract specific rows by metric name.
    """
    data: dict = {}

    def metric(name: str, default: float | None = None) -> float | None:
        """Match a table row: '  <name>  <unit>  <value>'."""
        # ncu table rows: leading spaces, metric name, whitespace, unit, whitespace, numeric value
        pattern = rf"^\s+{name}\s+\S+\s+([\d.]+)\s*$"
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return default

    # Speed of Light
    data["compute_throughput_pct"] = metric(r"Compute \(SM\) Throughput")
    data["sm_active_cycles"] = metric(r"SM Active Cycles")
    data["memory_throughput_pct"] = metric(r"Memory Throughput")
    data["dram_throughput_pct"] = metric(r"DRAM Throughput")
    data["l1_tex_throughput_pct"] = metric(r"L1/TEX Cache Throughput")
    data["l2_throughput_pct"] = metric(r"L2 Cache Throughput")
    sm_freq_ghz = metric(r"SM Frequency")
    data["sm_frequency_mhz"] = sm_freq_ghz * 1000 if sm_freq_ghz is not None else None

    # Occupancy
    data["theoretical_occupancy_pct"] = metric(r"Theoretical Occupancy")
    data["achieved_occupancy_pct"] = metric(r"Achieved Occupancy")
    data["regs_per_thread"] = metric(r"Registers Per Thread")
    data["block_limit_regs"] = metric(r"Block Limit Registers")
    data["block_limit_smem"] = metric(r"Block Limit Shared Mem")

    # Scheduler Stats
    data["no_eligible_pct"] = metric(r"No Eligible")
    data["one_or_more_eligible_pct"] = metric(r"One or More Eligible")
    data["active_warps_per_sched"] = metric(r"Active Warps Per Scheduler")
    data["eligible_warps_per_sched"] = metric(r"Eligible Warps Per Scheduler")
    data["issued_warp_per_sched"] = metric(r"Issued Warp Per Scheduler")

    # Warp State Stats
    data["warp_cycles_per_issued"] = metric(r"Warp Cycles Per Issued Instruction")

    # Top stall reason (narrative line, not a table row)
    stall_match = re.search(r"spends ([\d.]+) cycles being stalled (.+?)(?:\.\s|$)", text, re.DOTALL)
    if stall_match:
        data["top_stall_cycles"] = float(stall_match.group(1))
        data["top_stall_reason"] = stall_match.group(2).strip()

    # Estimated speedup (narrative)
    sp_match = re.search(r"Est\.\s*(?:Local\s*)?Speedup:\s*([\d.]+)%", text)
    if sp_match:
        data["est_speedup_stall_pct"] = float(sp_match.group(1))

    # L2 hit rate
    data["l2_hit_rate_pct"] = metric(r"L2 Hit")

    return data


def print_ncu_report(
    ncu: dict,
    gpu: dict,
    benchmark_peak_pct: float,
    theoretical_occ_pct: float,
) -> None:
    """Print ncu actual data alongside theoretical analysis + bottleneck summary."""
    print()
    print("  --- ncu Profiling (actual hardware data) ---")

    # Speed of Light
    ct = ncu.get("compute_throughput_pct")
    if ct is not None:
        gap = ct - benchmark_peak_pct
        print(f"  {'TC Throughput':<20} {ct:.2f}%   (benchmark: {benchmark_peak_pct:.1f}% → gap {gap:+.1f}%)")
    if ncu.get("memory_throughput_pct") is not None:
        print(f"  {'Memory Throughput':<20} {ncu['memory_throughput_pct']:.2f}%")
    if ncu.get("dram_throughput_pct") is not None:
        print(f"  {'DRAM Throughput':<20} {ncu['dram_throughput_pct']:.2f}%")
    if ncu.get("sm_frequency_mhz") is not None:
        print(f"  {'SM Frequency':<20} {ncu['sm_frequency_mhz']:.0f} MHz (boost: {gpu['clock_mhz']:.0f} MHz)")
    print()

    # Occupancy
    ao = ncu.get("achieved_occupancy_pct")
    to = ncu.get("theoretical_occupancy_pct")
    if ao is not None:
        theo = to if to is not None else theoretical_occ_pct
        gap = theo - ao
        print(f"  {'Achieved Occupancy':<20} {ao:.2f}%  (theoretical: {theo:.2f}% → gap {gap:.2f}%)")
    if ncu.get("regs_per_thread") is not None:
        regs = int(ncu["regs_per_thread"])
        print(f"  {'Registers':<20} {regs}/thread")
    print()

    # Scheduler
    ne = ncu.get("no_eligible_pct")
    if ne is not None:
        print(
            f"  {'No Eligible Warps':<20} {ne:.2f}%  ({'warps mostly stalled' if ne > 80 else 'some warp parallelism'})"
        )
    if ncu.get("eligible_warps_per_sched") is not None:
        print(f"  {'Eligible Warps/Sched':<20} {ncu['eligible_warps_per_sched']:.2f}")
    print()

    # Warp State / Stall Analysis
    wcpi = ncu.get("warp_cycles_per_issued")
    if wcpi is not None:
        print(f"  {'Cycles/Issued Inst':<20} {wcpi:.2f}")
    if ncu.get("top_stall_reason"):
        cycles = ncu.get("top_stall_cycles", 0)
        pct = cycles / wcpi * 100 if wcpi else 0
        reason = ncu["top_stall_reason"].split("\n")[0].strip()[:60]
        print(f"  {'Top Stall':<20} {reason} ({pct:.1f}%, {cycles:.1f} cycles)")
    print()

    # Bottleneck Summary
    print("  --- Bottleneck Summary ---")
    bottlenecks: list[tuple[str, float, str]] = []

    if ncu.get("top_stall_reason"):
        sp = ncu.get("est_speedup_stall_pct", 0)
        bottlenecks.append(
            (
                "CTA barrier stall",
                sp,
                f"pipeline mbarrier sync ({ncu['top_stall_reason']})",
            )
        )

    imb = ncu.get("est_speedup_imbalance_pct", 0)
    if imb > 0:
        bottlenecks.append(("SM load imbalance", imb, "grid tail → Split-K"))

    ct_val = ncu.get("compute_throughput_pct") or 0
    if ct_val > 0:
        gap = ct_val - benchmark_peak_pct
        if abs(gap) > 2:
            bottlenecks.append(
                (
                    "TC throughput gap",
                    abs(gap),
                    f"ncu {ct_val:.1f}% vs benchmark {benchmark_peak_pct:.1f}%",
                )
            )

    for i, (name, speedup, detail) in enumerate(bottlenecks, 1):
        label = "Primary" if i == 1 else f"#{i}"
        print(f"  {label:<20} {name} → {detail}")
        if speedup > 0:
            print(f"  {'':<20} est. speedup: {speedup:.1f}%")

    if not bottlenecks:
        print("  No significant bottlenecks detected.")


__all__ = [
    "PRINT_LENGTH",
    "KernelMeta",
    "compare_tensor",
    "cuda_bench",
    "get_gpu_info",
    "parse_ncu_output",
    "print_bench_report",
    "print_ncu_report",
    "relative_error",
    "run_ncu_profile",
    "theoretical_occupancy",
]
