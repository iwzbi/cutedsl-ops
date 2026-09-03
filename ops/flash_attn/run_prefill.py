"""Run + validate FlashAttention prefill (exercise 1: bf16 varlen multi-stage).

Usage::

    python ops/flash_attn/run_prefill.py                       # all shapes
    python ops/flash_attn/run_prefill.py --shapes 2,5          # only those shapes
    python ops/flash_attn/run_prefill.py --bench               # + TFLOPS report
    python ops/flash_attn/run_prefill.py --ncu                 # + ncu profiling

Shapes are varlen-form ``(H_q, H_kv, D, [seqlens...])``: the batch dimension
is implicit (B = len(seqlens)), every sequence is always causal, and the
harness pads each batch's flattened segment to a BLK_M=64 multiple (kernel
precondition).  ``pack_varlen`` builds the flattened inputs + ``cu_seqlens``.

The single shape list ``PREFILL_SHAPES`` (reference.py) drives both the
correctness gate and ``--bench`` — there is no separate bench-only set.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
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
    BLK_M,
    NUM_THREADS,
    FlashAttnPrefillBf16Multistage,
)
from ops.flash_attn.reference import PREFILL_SHAPES, allclose, pack_varlen, pick_split


ATOL = 0.016
STAGES = int(os.environ.get("FA_STAGES", "2"))  # K/V ring depth (v6: stages=2 wins)


def _split_for(shape):
    """Split-KV factor (grid z): FA_SPLIT env overrides for A/B benches, else
    auto by grid shape (v6: pick_split=2 wins on small grids)."""
    if "FA_SPLIT" in os.environ:
        return int(os.environ["FA_SPLIT"])
    H_q, _H_kv, _D, seqlens = shape
    return pick_split(((max(seqlens) + BLK_M - 1) // BLK_M) * len(seqlens) * H_q)


DESC = "bf16 multi-stage varlen (single WG, class-based)"


def _gen_bf16(B, H_q, H_kv, max_s, D, device="cuda"):
    torch.manual_seed(41)
    q = torch.randn(B, H_q, max_s, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, max_s, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, max_s, D, device=device, dtype=torch.bfloat16) * 0.5
    return q, k, v


def run_case(shape, *, bench=False, use_ncu=False) -> bool:
    """Run ex.1 varlen prefill on one varlen shape against torch SDPA."""
    H_q, H_kv, D, seqlens = shape
    B = len(seqlens)
    max_s = max(seqlens)
    split = _split_for(shape)
    q, k, v = _gen_bf16(B, H_q, H_kv, max_s, D)
    q_cat, k_cat, v_t, o_cat, seqlens_t, cu_seqlens = pack_varlen(q, k, v, seqlens)
    pad_offsets = cu_seqlens.cpu().numpy()
    # v_t's S axis is padded to a BLK_M multiple (pack_varlen); TMA V tiles
    # index it at BLK_N granularity so mirror that padded extent in the
    # kernel's max_seqlens (matches stride (S,1,D*S) of the V view).
    s_pad = v_t.shape[3]

    print(f"\n{'=' * 80}")
    print(f"  Exercise 1: {DESC}  | H_q={H_q} H_kv={H_kv} D={D} seqlens={seqlens}")
    print(f"{'=' * 80}")

    print("Compiling CuTe DSL (ex.1, bf16 varlen) ...")
    instance = FlashAttnPrefillBf16Multistage(num_stages=STAGES, split_k=split)
    t_pad = o_cat.shape[0]
    po = torch.empty(t_pad, H_q, split, D, device="cuda", dtype=torch.float32)
    pm = torch.empty(t_pad, H_q, split, device="cuda", dtype=torch.float32)
    pl = torch.empty(t_pad, H_q, split, device="cuda", dtype=torch.float32)
    compiled = cute.compile(
        instance,
        # q_cat: (T, H_q, D) bf16, strides (H_q*D, D, 1) — contiguous; T = Σ_b ceil(seq_b/64)*64
        make_cute_tensor(q_cat, leading_dim=2),
        # k_cat: (T, H_kv, D) bf16, strides (H_kv*D, D, 1) — contiguous
        make_cute_tensor(k_cat, leading_dim=2),
        # v_t: (B, H_kv, D, S) bf16, strides (H_kv*D*S, D*S, S, 1) — K-major for PV MMA
        make_cute_tensor(v_t, leading_dim=3),
        # o_cat: (T, H_q, D) bf16, strides (H_q*D, D, 1) — zero-filled output
        make_cute_tensor(o_cat, leading_dim=2),
        # seqlens_t: (B,) int32, real per-batch lengths (not padded)
        make_cute_tensor(seqlens_t, leading_dim=0),
        # cu_seqlens: (B+1,) int32, padded 64-aligned cumulative offsets
        make_cute_tensor(cu_seqlens, leading_dim=0),
        # split-KV workspace (fp32): partial O (T,H_q,S,D) + per-split max/sum
        # (T,H_q,S).  Unused when split==1 but still required by the signature.
        make_cute_tensor(po, leading_dim=3),
        make_cute_tensor(pm, leading_dim=2),
        make_cute_tensor(pl, leading_dim=2),
        make_stream(),
        s_pad,
        H_q,
        H_kv,
        D,
        options="--enable-tvm-ffi --generate-line-info",
    )
    compiled(q_cat, k_cat, v_t, o_cat, seqlens_t, cu_seqlens, po, pm, pl)
    torch.cuda.synchronize()

    # Per-batch comparison at real lengths (varlen, always causal).
    kk = k.repeat_interleave(H_q // H_kv, dim=1)
    vv = v.repeat_interleave(H_q // H_kv, dim=1)
    ref = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
    ok = True
    for b in range(B):
        p0 = int(pad_offsets[b])
        seg = o_cat[p0 : p0 + seqlens[b]]
        ref_b = ref[b, :, : seqlens[b], :].permute(1, 0, 2).contiguous()
        name = f"ex1 varlen b{b} seq{seqlens[b]}"
        ok = allclose(ref_b, seg, atol=ATOL, name=name) and ok

    if bench and ok:
        ms = cuda_bench(compiled, q_cat, k_cat, v_t, o_cat, seqlens_t, cu_seqlens, po, pm, pl)
        # FLOPs use max_s x max_s (the padded causal problem; causal skips the
        # upper triangle so this overcounts — see PERFLOG).
        flops = int(4 * B * H_q * max_s * max_s * D)
        gmem_bytes = int(
            B * H_q * max_s * D * q.element_size()
            + B * H_kv * max_s * D * k.element_size()
            + B * H_kv * max_s * D * v.element_size()
            + B * H_q * max_s * D * q.element_size()
        )
        grid_m = (max_s + BLK_M - 1) // BLK_M
        grid_blocks = B * H_q * grid_m
        num_sms = get_gpu_info()["num_sms"]
        grid_blocks_min = min(grid_blocks, num_sms)

        meta = KernelMeta(
            name="FA-Prefill-ex1",
            tile_dims={"BLK_M": BLK_M, "BLK_N": 64, "D": D, "NUM_STAGES": STAGES, "SPLIT": split},
            block_threads=NUM_THREADS,
            block_description=DESC,
            grid_mode="standard",
        )
        print_bench_report(
            ms=ms,
            problem_shape=(B, H_q, max_s, max_s, D),
            dtype=q.dtype,
            flops=flops,
            gmem_bytes=gmem_bytes,
            ws_count=1,
            meta=meta,
            grid_blocks=grid_blocks_min,
        )

    if use_ncu and ok:
        program_cmd = [sys.executable, os.path.abspath(__file__), "--ncu"]
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

    argv = sys.argv[1:]
    bench = "--bench" in argv
    use_ncu = "--ncu" in argv

    shape_indices = None
    if "--shapes" in argv:
        idx = argv.index("--shapes") + 1
        shape_indices = [int(x) for x in argv[idx].split(",")]
    shapes = list(PREFILL_SHAPES)
    if shape_indices is not None:
        shapes = [shapes[i] for i in shape_indices]

    if os.environ.get("NCU_PROFILING") == "1":
        run_case(PREFILL_SHAPES[0])
        return

    counters = {"succeed": 0, "failed": 0}
    for shape in shapes:
        if run_case(shape, bench=bench, use_ncu=use_ncu):
            counters["succeed"] += 1
        else:
            counters["failed"] += 1

    print(f"\n Summary: {counters['succeed']} succeed, {counters['failed']} failed ".center(PRINT_LENGTH, "="))
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
