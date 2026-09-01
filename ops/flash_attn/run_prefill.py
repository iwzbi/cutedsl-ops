"""Run + validate FlashAttention prefill exercises (1, 2, 4).

Usage::

    python ops/flash_attn/run_prefill.py                       # all exercises
    python ops/flash_attn/run_prefill.py --ex 1                 # only exercise 1
    python ops/flash_attn/run_prefill.py --ex 1 --bench         # + TFLOPS report
    python ops/flash_attn/run_prefill.py --ex 4 --ncu           # + ncu profiling
    python ops/flash_attn/run_prefill.py --ex 2 --causal        # causal mask

Exercises:
  1 — ``prefill_bf16_multistage``: bf16, single-WG, kStage=2
  2 — ``prefill_bf16_warpspec``:   bf16, warp-specialized (1 DMA + 2 MMA WGs)
  4 — ``prefill_fp8``:              fp8, paged KV, per-tensor KV scales

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
    KernelMeta,
    cuda_bench,
    get_gpu_info,
    parse_ncu_output,
    print_bench_report,
    run_ncu_profile,
)
from common.cute_runtime import make_cute_tensor, make_stream
from ops.flash_attn.kernels.prefill_bf16_multistage import (
    BLK_M as EX1_BLK_M,
)
from ops.flash_attn.kernels.prefill_bf16_multistage import (
    NUM_THREADS as EX1_THREADS,
)
from ops.flash_attn.kernels.prefill_bf16_multistage import (
    FlashAttnPrefillBf16Multistage,
)
from ops.flash_attn.kernels.prefill_bf16_warpspec import (
    BLK_M as EX2_BLK_M,
)
from ops.flash_attn.kernels.prefill_bf16_warpspec import (
    NUM_STAGES as EX2_STAGES,
)
from ops.flash_attn.kernels.prefill_bf16_warpspec import (
    NUM_THREADS as EX2_THREADS,
)
from ops.flash_attn.kernels.prefill_bf16_warpspec import (
    flash_attn_prefill_bf16_warpspec as ex2_fn,
)
from ops.flash_attn.kernels.prefill_fp8 import (
    BLK_M as EX4_BLK_M,
)
from ops.flash_attn.kernels.prefill_fp8 import (
    NUM_STAGES as EX4_STAGES,
)
from ops.flash_attn.kernels.prefill_fp8 import (
    NUM_THREADS as EX4_THREADS,
)
from ops.flash_attn.kernels.prefill_fp8 import (
    flash_attn_prefill_fp8 as ex4_fn,
)
from ops.flash_attn.reference import (
    allclose,
    ref_prefill_bf16,
    ref_prefill_fp8,
)


# Default prefill shapes: (B, H, H_kv, M, N, D)
DEFAULT_SHAPES = [
    (1, 1, 1, 64, 64, 64),
    (1, 1, 1, 512, 512, 64),
]

EXERCISES = {
    1: {
        "cls": FlashAttnPrefillBf16Multistage,
        "dtype": "bf16",
        "atol": 0.016,
        "paged": False,
        "blk_m": EX1_BLK_M,
        "stages": 2,
        "threads": EX1_THREADS,
        "desc": "bf16 multi-stage (single WG, class-based)",
        "v_natural": False,
    },
    2: {
        "fn": ex2_fn,
        "dtype": "bf16",
        "atol": 0.016,
        "paged": False,
        "blk_m": EX2_BLK_M,
        "stages": EX2_STAGES,
        "threads": EX2_THREADS,
        "desc": "bf16 warp-spec (1 DMA + 2 MMA)",
    },
    4: {
        "fn": ex4_fn,
        "dtype": "fp8",
        "atol": 0.05,
        "paged": True,
        "blk_m": EX4_BLK_M,
        "stages": EX4_STAGES,
        "threads": EX4_THREADS,
        "desc": "fp8 paged (per-tensor KV scale)",
    },
}


def _gen_bf16(B, H, H_kv, M, N, D, device="cuda"):
    torch.manual_seed(41)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, M, D, device=device, dtype=torch.bfloat16) * (scale**0.5)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.bfloat16) * (scale**0.5)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.bfloat16) * 0.5
    o = torch.zeros(B, H, M, D, device=device, dtype=torch.bfloat16)
    return q, k, v, o


def _gen_fp8_paged(B, H, H_kv, M, N, D, page_size=64, device="cuda"):
    torch.manual_seed(10086)
    q = torch.randn(B, H, M, D, device=device, dtype=torch.float8_e4m3fn)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float8_e4m3fn)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float8_e4m3fn)
    o = torch.zeros(B, H, M, D, device=device, dtype=torch.bfloat16)

    # Per-token-per-head Q scale, per-tensor K/V scales
    q_scale = torch.rand(B, H, M, 1, device=device, dtype=torch.float32) * 0.5 + 0.5
    k_scale = float(torch.rand(1, device=device).item() * 0.5 + 0.5)
    v_scale = float(torch.rand(1, device=device).item() * 0.5 + 0.5)

    # Convert to paged layout: (num_pages, H_kv, page_size, D)
    num_pages = (N + page_size - 1) // page_size * B
    k_pages = torch.randn(num_pages, H_kv, page_size, D, device=device, dtype=torch.float8_e4m3fn)
    v_pages = torch.randn(num_pages, H_kv, page_size, D, device=device, dtype=torch.float8_e4m3fn)

    # Fill pages from the dense tensors
    for b in range(B):
        for p in range((N + page_size - 1) // page_size):
            page_idx = b * ((N + page_size - 1) // page_size) + p
            start = p * page_size
            end = min(start + page_size, N)
            k_pages[page_idx, :, : end - start, :] = k[b, :, start:end, :]
            v_pages[page_idx, :, : end - start, :] = v[b, :, start:end, :]

    # Block table: (B, max_blocks), -1 for padding
    max_blocks = (N + page_size - 1) // page_size
    block_table = torch.arange(max_blocks, device=device, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()
    for b in range(1, B):
        block_table[b] = block_table[0] + b * max_blocks

    return q, k_pages, v_pages, block_table, o, q_scale, k_scale, v_scale


def run_case(
    ex_num: int,
    shape: tuple[int, ...],
    *,
    is_causal: bool = False,
    bench: bool = False,
    use_ncu: bool = False,
) -> bool:
    ex = EXERCISES[ex_num]
    B, H, H_kv, M, N, D = shape
    BH = B * H
    scale = 1.0 / math.sqrt(D)

    print(f"\n{'=' * 80}")
    print(f"  Exercise {ex_num}: {ex['desc']}  | B={B} H={H} H_kv={H_kv} M={M} N={N} D={D} causal={is_causal}")
    print(f"{'=' * 80}")

    if ex["dtype"] == "bf16":
        q, k, v, o = _gen_bf16(B, H, H_kv, M, N, D)
        q3 = q.view(BH, M, D)
        k3 = k.view(B * H_kv, N, D)
        o3 = o.view(BH, M, D)

        if ex.get("v_natural", False):
            v3 = v.view(B * H_kv, N, D)
        else:
            v3 = v.view(B * H_kv, N, D).transpose(1, 2).contiguous()

        print(f"Compiling CuTe DSL (ex.{ex_num}, bf16) ...")
        if "cls" in ex:
            instance = ex["cls"]()
            compiled = cute.compile(
                instance,
                make_cute_tensor(q3, leading_dim=2),
                make_cute_tensor(k3, leading_dim=2),
                make_cute_tensor(v3, leading_dim=2),
                make_cute_tensor(o3, leading_dim=2),
                make_stream(),
                BH,
                M,
                N,
                D,
                options="--enable-tvm-ffi --generate-line-info",
            )
        else:
            compiled = cute.compile(
                ex["fn"],
                make_cute_tensor(q3, leading_dim=2),
                make_cute_tensor(k3, leading_dim=2),
                make_cute_tensor(v3, leading_dim=2),
                make_cute_tensor(o3, leading_dim=2),
                make_stream(),
                is_causal,
                BH,
                M,
                N,
                D,
                options="--enable-tvm-ffi --generate-line-info",
            )
        compiled(q3, k3, v3, o3)
        torch.cuda.synchronize()

        ref = ref_prefill_bf16(q, k, v, is_causal=is_causal, scale=scale)
        ok = allclose(ref, o, atol=ex["atol"], name=f"ex{ex_num} bf16 {'causal' if is_causal else ''}")

    elif ex["dtype"] == "fp8":
        page_size = 64
        q, k_pages, v_pages, block_table, o, q_scale, k_scale, v_scale = _gen_fp8_paged(B, H, H_kv, M, N, D, page_size)
        q3 = q.view(BH, M, D)
        o3 = o.view(BH, M, D)

        print(f"Compiling CuTe DSL (ex.{ex_num}, fp8, paged) ...")
        compiled = cute.compile(
            ex["fn"],
            make_cute_tensor(q3, leading_dim=2),
            make_cute_tensor(k_pages, leading_dim=3),
            make_cute_tensor(v_pages, leading_dim=3),
            make_cute_tensor(block_table, leading_dim=1),
            make_cute_tensor(o3, leading_dim=2),
            make_cute_tensor(q_scale.view(BH, M, 1), leading_dim=2),
            make_stream(),
            k_scale,
            v_scale,
            is_causal,
            BH,
            M,
            N,
            D,
            page_size,
            options="--enable-tvm-ffi --generate-line-info",
        )
        compiled(q3, k_pages, v_pages, block_table, o3, q_scale.view(BH, M, 1))
        torch.cuda.synchronize()

        ref = ref_prefill_fp8(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        ok = allclose(ref, o, atol=ex["atol"], name=f"ex{ex_num} fp8 {'causal' if is_causal else ''}")
    else:
        raise ValueError(f"Unknown dtype: {ex['dtype']}")

    if bench and ok:
        if ex["dtype"] == "bf16":
            ms = cuda_bench(compiled, q3, k3, v3, o3)
        else:
            ms = cuda_bench(compiled, q3, k_pages, v_pages, block_table, o3, q_scale.view(BH, M, 1))

        flops = int(4 * B * H * M * N * D)  # 2*(QK^T) + 2*(PV)
        gmem_bytes = int(
            B * H * M * D * q.element_size()
            + B * H_kv * N * D * k.element_size()
            + B * H_kv * N * D * v.element_size()
            + B * H * M * D * o.element_size()
        )
        grid_m = (M + ex["blk_m"] - 1) // ex["blk_m"]
        grid_blocks = BH * grid_m
        num_sms = get_gpu_info()["num_sms"]
        grid_blocks_min = min(grid_blocks, num_sms)

        meta = KernelMeta(
            name=f"FA-Prefill-ex{ex_num}",
            tile_dims={"BLK_M": ex["blk_m"], "BLK_N": 64, "D": D, "NUM_STAGES": ex["stages"]},
            block_threads=ex["threads"],
            block_description=ex["desc"],
            grid_mode="persistent" if ex_num >= 2 else "standard",
        )
        print_bench_report(
            ms=ms,
            problem_shape=(B, H, M, N, D),
            dtype=q.dtype,
            flops=flops,
            gmem_bytes=gmem_bytes,
            ws_count=1,
            meta=meta,
            grid_blocks=grid_blocks_min,
        )

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
    exercises = [int(a.split("=")[1]) for a in ex_args] if ex_args else [1, 2, 4]
    bench = "--bench" in sys.argv
    use_ncu = "--ncu" in sys.argv
    is_causal = "--causal" in sys.argv
    sys.argv = [a for a in sys.argv if not a.startswith("--")]

    if os.environ.get("NCU_PROFILING") == "1":
        for ex_num in exercises:
            run_case(ex_num, DEFAULT_SHAPES[0])
        return

    counters = {"succeed": 0, "failed": 0}
    for ex_num in exercises:
        for shape in DEFAULT_SHAPES:
            if run_case(ex_num, shape, is_causal=is_causal, bench=bench, use_ncu=use_ncu):
                counters["succeed"] += 1
            else:
                counters["failed"] += 1

    print(f"\n Summary: {counters['succeed']} succeed, {counters['failed']} failed ".center(PRINT_LENGTH, "="))
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
