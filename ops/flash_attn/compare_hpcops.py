"""Compare cutedsl ex.1 varlen FlashAttention vs hpc-ops attention_prefill_bf16.

Both produce causal varlen prefill output.  We compare:
1. **Precision**: torch SDPA vs hpc-ops vs cutedsl — 3-way agreement per batch.
2. **Performance**: TFLOPS and ms/call for both across shared shapes.

All shapes are varlen-form ``(H_q, H_kv, D, [seqlens...])``: the batch
dimension is implicit (B = len(seqlens)) and each batch is a causal
self-attention sequence of its own length.  hpc-ops runs on the real-length
flattening with real ``cu_seqlens``; cutedsl runs on the BLK_M=64-padded
flattening (kernel precondition, packed by ``pack_varlen``) with padded
``cu_seqlens``.  The comparison always slices each implementation back to the
real-length per-batch segments, so the 3-way agreement is on identical logical
tensors.

Usage::
    python ops/flash_attn/compare_hpcops.py                 # all shapes
    python ops/flash_attn/compare_hpcops.py --shapes 2      # only shape index 2

Shapes come from the single authority ``PREFILL_SHAPES`` (reference.py),
shared with run_prefill.py — the same list drives correctness and benchmark.
"""

from __future__ import annotations

import os
import sys

import hpc
import torch
import torch.nn.functional as F
from cutlass import cute


sys.path.insert(0, ".")
from common.bench import cuda_bench, get_gpu_info
from common.cute_runtime import make_cute_tensor, make_stream
from ops.flash_attn.kernels.prefill_bf16_multistage import (
    BLK_M,
    FlashAttnPrefillBf16Multistage,
)
from ops.flash_attn.reference import PREFILL_SHAPES, allclose, pack_varlen, pick_split


def _gen_data(H_q, H_kv, seqlens, D, device="cuda"):
    """Generate shared Q/K/V data (unscaled randn, v*0.5)."""
    torch.manual_seed(41)
    max_s = max(seqlens)
    q = torch.randn(len(seqlens), H_q, max_s, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(len(seqlens), H_kv, max_s, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(len(seqlens), H_kv, max_s, D, device=device, dtype=torch.bfloat16) * 0.5
    return q, k, v


def _flatten_padded(t, seqlens, max_s):
    """Padded varlen flatten: (B, H, S, D) -> (B*max_s, H, D).

    Each batch occupies exactly ``max_s`` rows (rows beyond ``seqlens[b]`` are
    zeros; causal masking means they never contribute).  hpc-ops expects this
    fixed-stride layout: its ``cu_seqlens`` are evenly spaced ``[0, max_s, ...]``.
    """
    tn = t.permute(0, 2, 1, 3)  # (B, S, H, D)
    return tn.reshape(-1, t.shape[1], t.shape[3]).contiguous()


def _sdpa_varlen_ref(q, k, v, seqlens):
    """Causal attention reference via torch SDPA, sliced to real lengths.

    SDPA runs on the full padded (B, H, max_s, D) tensors with is_causal=True;
    rows beyond seqlens[b] are masked (upper-triangle) and never read back, so
    slicing [b, :, :seqlens[b]] yields the exact varlen result.  GQA handled by
    repeat_interleave (SDPA's built-in GQA support is version-dependent).
    """
    gqa = q.shape[1] // k.shape[1]
    kk = k.repeat_interleave(gqa, dim=1)
    vv = v.repeat_interleave(gqa, dim=1)
    out = F.scaled_dot_product_attention(q, kk, vv, is_causal=True).to(torch.bfloat16)
    segs = [out[b, :, : seqlens[b], :].permute(1, 0, 2).contiguous() for b in range(len(seqlens))]
    return torch.cat(segs, dim=0)  # (total_real, H, D)


def _run_hpcops(q, k, v, seqlens, H_q, D):
    """Run hpc-ops varlen prefill (always causal). Returns (flat_out, ms).

    hpc-ops applies 1/sqrt(D) softmax scale internally.  We pass raw unscaled
    q/k to match our cutedsl kernel's semantics (pre-scaling would double it).
    Its input layout is padded: each batch occupies exactly max_s rows (see
    _flatten_padded).  We slice the output back to real lengths and
    concatenate in the same (total_real, H, D) segment order as the refs.
    """
    max_s = max(seqlens)
    q_flat = _flatten_padded(q, seqlens, max_s)
    k_flat = _flatten_padded(k, seqlens, max_s)
    v_flat = _flatten_padded(v, seqlens, max_s)

    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(len(seqlens) + 1, dtype=torch.int32, device="cuda") * max_s

    out_pad = hpc.attention_prefill_bf16(q_flat, k_flat, v_flat, seqlens_t, cu_seqlens, max_s)

    def _call():
        hpc.attention_prefill_bf16(q_flat, k_flat, v_flat, seqlens_t, cu_seqlens, max_s, out_pad)

    ms = cuda_bench(_call)

    segs = [out_pad[b * max_s : b * max_s + seqlens[b]] for b in range(len(seqlens))]
    flat = torch.cat(segs, dim=0).contiguous()  # (total_real, H, D)
    return flat, ms


def _run_cutedsl(q, k, v, seqlens, H_q, H_kv, D, split=1):
    """Run cutedsl ex.1 varlen prefill on the SAME logical inputs as hpc-ops.

    Inputs are the BLK_M=64-padded flatten + padded cu_seqlens (kernel
    precondition); output is sliced back to real lengths and concatenated in
    the same (total_real, H, D) segment order.  `split` is the split-KV factor
    (grid z); split=1 keeps the fused epilogue and the workspace is unused.
    """
    q_cat, k_cat, v_t, o_cat, seqlens_t, cu_seqlens = pack_varlen(q, k, v, seqlens)
    pad_offsets = cu_seqlens.cpu().numpy()

    t_pad = o_cat.shape[0]
    po = torch.empty(t_pad, H_q, split, D, device="cuda", dtype=torch.float32)
    pm = torch.empty(t_pad, H_q, split, device="cuda", dtype=torch.float32)
    pl = torch.empty(t_pad, H_q, split, device="cuda", dtype=torch.float32)

    instance = FlashAttnPrefillBf16Multistage(num_stages=int(os.environ.get("FA_STAGES", "2")), split_k=split)
    compiled = cute.compile(
        instance,
        make_cute_tensor(q_cat, leading_dim=2),
        make_cute_tensor(k_cat, leading_dim=2),
        make_cute_tensor(v_t, leading_dim=3),
        make_cute_tensor(o_cat, leading_dim=2),
        make_cute_tensor(seqlens_t, leading_dim=0),
        make_cute_tensor(cu_seqlens, leading_dim=0),
        make_cute_tensor(po, leading_dim=3),
        make_cute_tensor(pm, leading_dim=2),
        make_cute_tensor(pl, leading_dim=2),
        make_stream(),
        v_t.shape[3],
        H_q,
        H_kv,
        D,
        options="--enable-tvm-ffi --generate-line-info",
    )
    compiled(q_cat, k_cat, v_t, o_cat, seqlens_t, cu_seqlens, po, pm, pl)
    torch.cuda.synchronize()

    segs = [o_cat[int(pad_offsets[b]) : int(pad_offsets[b]) + seqlens[b]] for b in range(len(seqlens))]
    flat = torch.cat(segs, dim=0).contiguous()  # (total_real, H, D)

    def _call():
        compiled(q_cat, k_cat, v_t, o_cat, seqlens_t, cu_seqlens, po, pm, pl)

    ms = cuda_bench(_call)
    return flat, ms


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA GPU.")

    gpu = get_gpu_info()
    print(f"\n{'=' * 100}")
    print("  FlashAttention Prefill Comparison: cutedsl ex.1 (varlen) vs hpc-ops")
    print(
        f"  GPU: {gpu['name']} sm_{gpu['compute_capability'][-2:]}, {gpu['num_sms']} SMs, {gpu['peak_fp16_tflops']:.0f}T FP16 peak"
    )
    print("  All shapes are CAUSAL (hpc-ops is always causal), varlen-form (H_q,H_kv,D,seqlens)")
    print(f"{'=' * 100}")

    shape_indices = None
    if "--shapes" in sys.argv:
        idx = sys.argv.index("--shapes") + 1
        shape_indices = [int(x) for x in sys.argv[idx].split(",")]
    shapes = PREFILL_SHAPES if shape_indices is None else [PREFILL_SHAPES[i] for i in shape_indices]

    split_override = None
    if "FA_SPLIT" in os.environ:
        split_override = int(os.environ["FA_SPLIT"])
    if "--split" in sys.argv:
        split_override = int(sys.argv[sys.argv.index("--split") + 1])
    print(f"  cutedsl split_k = {split_override or 'auto (pick_split by grid)'}")

    print(f"\n{'Shape':<40} {'H':>3} {'Hkv':>4} {'D':>4} {'B':>3} {'max_s':>6}")
    print("-" * 100)

    for shape in shapes:
        H_q, H_kv, D, seqlens = shape
        print(f"  ({H_q},{H_kv},{D},{seqlens}){'':<8} {H_q:>3} {H_kv:>4} {D:>4} {len(seqlens):>3} {max(seqlens):>6}")

    print(f"\n{'=' * 100}")
    print(f"{'Shape':<40} {'Kernel':<12} {'ms/call':>10} {'TFLOPS':>10} {'peak%':>7}")
    print(f"{'-' * 100}")

    for shape in shapes:
        H_q, H_kv, D, seqlens = shape
        # Real attention work: sum of per-batch causal s² (hpc-ops and cutedsl
        # both roughly do this; padded rows of the cutedsl kernel are masked).
        flops = int(4 * H_q * sum(s * s for s in seqlens) * D)
        label = f"({H_q},{H_kv},{D},{seqlens})"
        # v6a: auto split-K by grid shape unless FA_SPLIT/--split overrides.
        split = split_override or pick_split(((max(seqlens) + BLK_M - 1) // BLK_M) * len(seqlens) * H_q)

        q, k, v = _gen_data(H_q, H_kv, seqlens, D)

        # torch reference (per-batch real-length segments, causal)
        ref = _sdpa_varlen_ref(q, k, v, seqlens)

        # hpc-ops
        hpc_out, hpc_ms = _run_hpcops(q, k, v, seqlens, H_q, D)
        hpc_tflops = flops / hpc_ms / 1e9
        hpc_peak_pct = hpc_tflops / gpu["peak_fp16_tflops"] * 100

        # cutedsl ex.1 (varlen, padded inputs — precision compared on real segs)
        cute_out, cute_ms = _run_cutedsl(q, k, v, seqlens, H_q, H_kv, D, split)
        cute_tflops = flops / cute_ms / 1e9
        cute_peak_pct = cute_tflops / gpu["peak_fp16_tflops"] * 100

        # 3-way agreement on identical (total_real, H, D) tensors
        allclose(ref, hpc_out, atol=0.016, name=f"{label} torch-vs-hpc")
        allclose(ref, cute_out, atol=0.016, name=f"{label} torch-vs-cute")
        allclose(hpc_out, cute_out, atol=0.016, name=f"{label} hpc-vs-cute")

        print(f"{label:<40} {'hpc-ops':<12} {hpc_ms:>10.3f} {hpc_tflops:>10.1f} {hpc_peak_pct:>6.1f}%")
        print(f"{'':<40} {'cutedsl':<12} {cute_ms:>10.3f} {cute_tflops:>10.1f} {cute_peak_pct:>6.1f}%")
        # ratio > 1 means cutedsl is FASTER than hpc-ops (hpc_ms / cute_ms)
        print(f"{'':<40} {'vs hpc':<12} {hpc_ms / cute_ms:>10.2f}x")
        print()

    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
