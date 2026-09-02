"""Quick varlen kernel smoke test: compile + run vs torch SDPA ref (causal).

Mirrors hpc-ops attention_prefill_bf16 usage. Tests variable-length batches,
including misaligned lengths (the harness pads each batch's flattened Q/K/O
segment to a BLK_M=64 multiple — required by the kernel; see kernel docstring).
"""

import math
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F
from cutlass import cute

from common.cute_runtime import make_cute_tensor, make_stream
from ops.flash_attn.kernels.prefill_bf16_multistage import FlashAttnPrefillBf16Multistage
from ops.flash_attn.reference import allclose, pack_varlen


SPLIT = int(os.environ.get("FA_SPLIT", "1"))


def run(B, H_q, H_kv, seqlens_list, D, pad_to=None):
    torch.manual_seed(41)
    scale_sqrt = (1.0 / math.sqrt(D)) ** 0.5
    max_s = max(seqlens_list)
    pad_to = pad_to or max_s

    q = torch.randn(B, H_q, pad_to, D, device="cuda", dtype=torch.bfloat16) * scale_sqrt
    k = torch.randn(B, H_kv, pad_to, D, device="cuda", dtype=torch.bfloat16) * scale_sqrt
    v = torch.randn(B, H_kv, pad_to, D, device="cuda", dtype=torch.bfloat16) * 0.5

    q_cat, k_cat, v_t, o_cat, seqlens, cu_seqlens = pack_varlen(q, k, v, seqlens_list)
    # Padded offsets for slicing the (real-length) result segments back out.
    padded_offsets = cu_seqlens.cpu().to(torch.int64).numpy()

    t_pad = o_cat.shape[0]
    po = torch.empty(t_pad, H_q, SPLIT, D, device="cuda", dtype=torch.float32)
    pm = torch.empty(t_pad, H_q, SPLIT, device="cuda", dtype=torch.float32)
    pl = torch.empty(t_pad, H_q, SPLIT, device="cuda", dtype=torch.float32)

    inst = FlashAttnPrefillBf16Multistage(split_k=SPLIT)
    print(f"Compiling varlen(B={B},H_q={H_q},H_kv={H_kv},seq={seqlens_list},max={pad_to},D={D},split={SPLIT}) ...")
    compiled = cute.compile(
        inst,
        make_cute_tensor(q_cat, leading_dim=2),
        make_cute_tensor(k_cat, leading_dim=2),
        make_cute_tensor(v_t, leading_dim=3),  # (B, H_kv, D, pad) pad contiguous
        make_cute_tensor(o_cat, leading_dim=2),  # (padded_total, H_q, D)
        make_cute_tensor(seqlens, leading_dim=0),
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
    compiled(q_cat, k_cat, v_t, o_cat, seqlens, cu_seqlens, po, pm, pl)
    torch.cuda.synchronize()

    kk = k.repeat_interleave(H_q // H_kv, dim=1)
    vv = v.repeat_interleave(H_q // H_kv, dim=1)
    ref = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
    ok = True
    for b in range(B):
        p0 = int(padded_offsets[b])
        seg = o_cat[p0 : p0 + seqlens_list[b]]  # (len, H_q, D)
        ref_b = ref[b, :, : seqlens_list[b], :].permute(1, 0, 2).contiguous()  # (len, H_q, D)
        name = f"varlen B{B}b{b}H{H_q}Hkv{H_kv}seq{seqlens_list[b]}D{D}"
        ok = allclose(ref_b, seg, atol=0.016, name=name) and ok
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    ok1 = run(1, 4, 4, [512], 128)
    ok2 = run(2, 4, 4, [512, 768], 128)
    ok3 = run(2, 4, 1, [256, 384], 128)
    ok4 = run(2, 4, 4, [200, 328], 128)  # misaligned: neither is a 64-multiple
    ok5 = run(3, 2, 1, [100, 240, 340], 128)  # misaligned + GQA + 3 batches
    print(f"\nSummary: {sum([ok1, ok2, ok3, ok4, ok5])}/5 passed")
    raise SystemExit(0 if (ok1 and ok2 and ok3 and ok4 and ok5) else 1)
