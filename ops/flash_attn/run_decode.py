"""Run + validate FlashAttention decode exercises (3, 5).

Usage::

    python ops/flash_attn/run_decode.py                  # all exercises
    python ops/flash_attn/run_decode.py --ex 3           # only exercise 3
    python ops/flash_attn/run_decode.py --ex 5 --bench   # + latency report
    python ops/flash_attn/run_decode.py --ex 3 --ncu     # + ncu profiling

Exercises:
  3 — ``decode_bf16_splitk``:  bf16, split-K + paged KV + LSE combine
  5 — ``decode_fp8``:          fp8, split-K + paged KV + Px256 fp8 quant

Decode is latency-bound (small M, large N).  The harness reports **median
latency in µs** (not TFLOPS) plus speedup vs torch SDPA baseline.

Until the kernel TODOs are implemented this harness prints ``Failed``.
"""

from __future__ import annotations

import math
import os
import sys

import torch
from cutlass import cute


sys.path.insert(0, ".")
from common.bench import (
    PRINT_LENGTH,
    cuda_bench,
    get_gpu_info,
    parse_ncu_output,
    run_ncu_profile,
)
from common.cute_runtime import make_cute_tensor, make_stream
from ops.flash_attn.kernels.decode_bf16_splitk import (
    BLK_M as EX3_BLK_M,
)
from ops.flash_attn.kernels.decode_bf16_splitk import (
    NUM_STAGES as EX3_STAGES,
)
from ops.flash_attn.kernels.decode_bf16_splitk import (
    NUM_THREADS as EX3_THREADS,
)
from ops.flash_attn.kernels.decode_bf16_splitk import (
    flash_attn_decode_bf16_splitk as ex3_fn,
)
from ops.flash_attn.kernels.decode_fp8 import (
    BLK_M as EX5_BLK_M,
)
from ops.flash_attn.kernels.decode_fp8 import (
    NUM_STAGES as EX5_STAGES,
)
from ops.flash_attn.kernels.decode_fp8 import (
    NUM_THREADS as EX5_THREADS,
)
from ops.flash_attn.kernels.decode_fp8 import (
    flash_attn_decode_fp8 as ex5_fn,
)
from ops.flash_attn.reference import (
    allclose,
    lse_combine,
    ref_decode_bf16,
    ref_decode_fp8,
)


# (B, H, H_kv, M, N, D, page_size)
DEFAULT_CASES = [
    (1, 1, 1, 1, 512, 128, 64),
    (1, 4, 1, 1, 1024, 128, 64),
    (16, 8, 1, 1, 2048, 128, 64),
    (16, 4, 4, 1, 4096, 128, 64),
]

EXERCISES = {
    3: {
        "fn": ex3_fn,
        "dtype": "bf16",
        "atol": 0.016,
        "kSplitK": 4,
        "blk_m": EX3_BLK_M,
        "stages": EX3_STAGES,
        "threads": EX3_THREADS,
        "desc": "bf16 split-K decode",
    },
    5: {
        "fn": ex5_fn,
        "dtype": "fp8",
        "atol": 0.1,
        "kSplitK": 4,
        "blk_m": EX5_BLK_M,
        "stages": EX5_STAGES,
        "threads": EX5_THREADS,
        "desc": "fp8 split-K decode (QK=SS SV=RS)",
    },
}

PAGE_SIZE = 64


def _gen_paged_kv(B, H, H_kv, M, N, D, dtype_str, device="cuda"):
    """Generate Q + paged KV cache + block_table."""
    if dtype_str == "bf16":
        torch.manual_seed(41)
        q = torch.randn(B, H, M, D, device=device, dtype=torch.bfloat16) * (1.0 / math.sqrt(D)) ** 0.5
        k_pages = torch.randn(
            B * ((N + PAGE_SIZE - 1) // PAGE_SIZE), H_kv, PAGE_SIZE, D, device=device, dtype=torch.bfloat16
        )
        v_pages = torch.randn_like(k_pages)
    else:
        torch.manual_seed(10086)
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float8_e4m3fn)
        k_pages = torch.randn(
            B * ((N + PAGE_SIZE - 1) // PAGE_SIZE), H_kv, PAGE_SIZE, D, device=device, dtype=torch.float8_e4m3fn
        )
        v_pages = torch.randn_like(k_pages)

    max_blocks = (N + PAGE_SIZE - 1) // PAGE_SIZE
    block_table = torch.arange(max_blocks, device=device, dtype=torch.int32)
    block_table = block_table.unsqueeze(0).expand(B, -1).contiguous()
    for b in range(1, B):
        block_table[b] = block_table[0] + b * max_blocks

    return q, k_pages, v_pages, block_table


def run_case(
    ex_num: int,
    shape: tuple[int, ...],
    *,
    bench: bool = False,
    use_ncu: bool = False,
) -> bool:
    ex = EXERCISES[ex_num]
    B, H, H_kv, M, N, D, _ = shape
    BH = B * H
    kSplitK = ex["kSplitK"]
    split_size = (N + kSplitK - 1) // kSplitK
    scale = 1.0 / math.sqrt(D)

    print(f"\n{'=' * 80}")
    print(f"  Exercise {ex_num}: {ex['desc']}  | B={B} H={H} H_kv={H_kv} M={M} N={N} D={D} splitK={kSplitK}")
    print(f"{'=' * 80}")

    q, k_pages, v_pages, block_table = _gen_paged_kv(B, H, H_kv, M, N, D, ex["dtype"])

    q3 = q.view(BH, M, D)
    blk_m = ex["blk_m"]
    o_partial = torch.zeros(kSplitK, BH, blk_m, D, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.full((kSplitK, BH, blk_m), float("-inf"), device="cuda", dtype=torch.float32)
    # Pre-transpose V for bf16 kernel (K-major TMA): (num_pages, H_kv, page_size, D) -> (num_pages, H_kv, D, page_size)
    v_pages_t = v_pages.transpose(-2, -1).contiguous() if ex["dtype"] == "bf16" else v_pages

    if ex["dtype"] == "bf16":
        print(f"Compiling CuTe DSL (ex.{ex_num}, bf16, splitK={kSplitK}) ...")
        compiled = cute.compile(
            ex["fn"],
            make_cute_tensor(q3, leading_dim=2),
            make_cute_tensor(k_pages, leading_dim=3),
            make_cute_tensor(v_pages_t, leading_dim=3),
            make_cute_tensor(block_table, leading_dim=1),
            make_cute_tensor(o_partial.view(kSplitK, BH, blk_m, D), leading_dim=3),
            make_cute_tensor(lse_partial.view(kSplitK, BH, blk_m), leading_dim=2),
            make_stream(),
            PAGE_SIZE,
            N,
            split_size,
            kSplitK,
            M,
            D,
            options="--enable-tvm-ffi --generate-line-info",
        )
        compiled(q3, k_pages, v_pages_t, block_table, o_partial, lse_partial)
        torch.cuda.synchronize()

        o_combined = lse_combine(
            o_partial.view(kSplitK, B, H, blk_m, D)[:, :, :, :M, :].contiguous(),
            lse_partial.view(kSplitK, B, H, blk_m)[:, :, :, :M].contiguous(),
        )
        ref = ref_decode_bf16(q, k_pages, v_pages, block_table, scale=scale)
        ok = allclose(ref, o_combined, atol=ex["atol"], name=f"ex{ex_num} bf16 decode")

    elif ex["dtype"] == "fp8":
        q_scale = torch.rand(B, H, M, 1, device="cuda", dtype=torch.float32) * 0.5 + 0.5
        k_scale = float(torch.rand(1, device="cuda").item() * 0.5 + 0.5)
        v_scale = float(torch.rand(1, device="cuda").item() * 0.5 + 0.5)

        print(f"Compiling CuTe DSL (ex.{ex_num}, fp8, splitK={kSplitK}) ...")
        compiled = cute.compile(
            ex["fn"],
            make_cute_tensor(q3, leading_dim=2),
            make_cute_tensor(k_pages, leading_dim=3),
            make_cute_tensor(v_pages, leading_dim=3),
            make_cute_tensor(block_table, leading_dim=1),
            make_cute_tensor(o_partial.view(kSplitK, BH, blk_m, D), leading_dim=3),
            make_cute_tensor(lse_partial.view(kSplitK, BH, blk_m), leading_dim=2),
            make_cute_tensor(q_scale.view(BH, M, 1), leading_dim=2),
            make_stream(),
            k_scale,
            v_scale,
            PAGE_SIZE,
            N,
            split_size,
            kSplitK,
            M,
            D,
            options="--enable-tvm-ffi --generate-line-info",
        )
        compiled(q3, k_pages, v_pages, block_table, o_partial, lse_partial, q_scale.view(BH, M, 1))
        torch.cuda.synchronize()

        o_combined = lse_combine(
            o_partial.view(kSplitK, B, H, blk_m, D)[:, :, :, :M, :].contiguous(),
            lse_partial.view(kSplitK, B, H, blk_m)[:, :, :, :M].contiguous(),
        )
        ref = ref_decode_fp8(
            q,
            k_pages,
            v_pages,
            block_table,
            scale=scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        ok = allclose(ref, o_combined, atol=ex["atol"], name=f"ex{ex_num} fp8 decode")
    else:
        raise ValueError(f"Unknown dtype: {ex['dtype']}")

    if bench and ok:
        if ex["dtype"] == "bf16":
            ms = cuda_bench(compiled, q3, k_pages, v_pages_t, block_table, o_partial, lse_partial)
        else:
            ms = cuda_bench(
                compiled, q3, k_pages, v_pages, block_table, o_partial, lse_partial, q_scale.view(BH, M, 1)
            )

        us = ms * 1e3  # ms -> µs
        flops = int(4 * B * H * M * N * D)
        gmem_bytes = int(
            B * H * M * D * q.element_size()
            + B * H_kv * N * D * k_pages.element_size() / k_pages.shape[0]  # per-request
            + B * H_kv * N * D * v_pages.element_size() / v_pages.shape[0]
            + B * H * M * D * 2  # output bf16
        )
        gpu = get_gpu_info()
        sep = "=" * 80
        print(f"\n{sep}")
        print(f"  FA-Decode ex{ex_num}  B={B} H={H} H_kv={H_kv} M={M} N={N} D={D} splitK={kSplitK}  | {gpu['name']}")
        print(sep)
        print(f"  {'Latency':<24} {us:.1f} µs/call  (median, CUDA events)")
        tflops = flops / ms / 1e9
        peak = (
            gpu["peak_fp16_tflops"] if ex["dtype"] == "bf16" else gpu.get("peak_fp8_tflops", gpu["peak_fp16_tflops"])
        )
        print(
            f"  {'TFLOPS':<24} {tflops:,.1f} / {peak:,.0f} peak ({tflops / peak * 100:.1f}%) — decode is latency-bound"
        )
        print(f"  {'gmem I/O':<24} {gmem_bytes / 1e6:,.1f} MB")
        print(
            f"  {'Bandwidth':<24} {gmem_bytes / ms / 1e6:,.0f} GB/s ({gmem_bytes / ms / 1e6 / gpu['hbm_bw_gbs'] * 100:.1f}% of {gpu['hbm_bw_gbs']:,.0f} GB/s)"
        )
        print(f"  {'Block':<24} {ex['threads']} threads ({ex['desc']})")
        print(f"  {'Tile':<24} BLK_M={ex['blk_m']}, BLK_N=64, D={D}, stages={ex['stages']}")
        print(f"  {'Grid':<24} ({kSplitK}, {BH}, 1) = {kSplitK * BH} blocks")
        print(f"{sep}")

    if use_ncu and ok:
        program_cmd = [sys.executable, os.path.abspath(__file__), "--ex", str(ex_num)]
        ncu_text = run_ncu_profile("flash_attn", program_cmd)
        ncu_data = parse_ncu_output(ncu_text)
        if ncu_data:
            gpu = get_gpu_info()
            from common.bench import print_ncu_report

            print_ncu_report(ncu_data, gpu, 0.0, 0.0)

    return ok


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable GPU (sm_90 recommended).")

    ex_args = [a for a in sys.argv if a.startswith("--ex=")]
    exercises = [int(a.split("=")[1]) for a in ex_args] if ex_args else [3, 5]
    bench = "--bench" in sys.argv
    use_ncu = "--ncu" in sys.argv
    sys.argv = [a for a in sys.argv if not a.startswith("--")]

    if os.environ.get("NCU_PROFILING") == "1":
        for ex_num in exercises:
            run_case(ex_num, DEFAULT_CASES[0])
        return

    counters = {"succeed": 0, "failed": 0}
    for ex_num in exercises:
        for shape in DEFAULT_CASES:
            if run_case(ex_num, shape, bench=bench, use_ncu=use_ncu):
                counters["succeed"] += 1
            else:
                counters["failed"] += 1

    print(f"\n Summary: {counters['succeed']} succeed, {counters['failed']} failed ".center(PRINT_LENGTH, "="))
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
