# FlashAttention Optimization Journey

Hopper H20 (sm_90, 78 SM, 148 TFLOPS FP16 peak), CuTe DSL (nvidia-cutlass-dsl 4.7.0).
References: torch SDPA (`F.scaled_dot_product_attention`, fp32) and hpc-ops
(`attention_prefill_bf16`, Tencent, sm_90, always-causal varlen).

Each step links to ncu raw reports in [`ncu_reports/`](./ncu_reports/) when available.
Use `git diff <prev-tag>..<tag> -- ops/flash_attn/kernels/<kernel>.py` to see code changes.

> Status legend: ✅ verified correct · 🚧 scaffold (compiles-expected, untested) · ⏳ not started

---

## Master Performance Table

All numbers BF16 input / FP32 accumulator / BF16 output, **causal varlen**, L2-flushed CUDA Events
(our kernel + hpc-ops via `compare_hpcops.py`). TFLOPS = 4·H·Σs² / t.
NOTE: causal skips the upper-triangle so the 4·H·Σs² formula overcounts our work at
M≈N — hpc-ops reports >100% peak for exactly this reason. Our kernel does NOT yet skip
future KV blocks (that is the v3 causal early-exit step).

| Shape (H,H_kv,D,seqlens) | hpc-ops | ex.1 v1-varlen | Speedup vs hpc |
|---|---|---|---|
| (4,4,128,[512]) | 0.030 ms / 18.2T | 1.108 ms / 0.5T | 37x |
| (8,8,128,[1024]) | 0.053 / 81.7T | 2.844 / 1.5T | 54x |
| (4,1,128,[512]) GQA | 0.030 / 18.1T | 1.107 / 0.5T | 37x |
| (4,4,128,[2048,2048]) | 0.095 / 180.7T | 11.49 / 1.5T | 121x |
| (8,2,128,[4096]) GQA | 0.286 / 240.4T | 37.6 / 1.8T | 132x |
| (8,8,128,[2048,2048,2048,2048]) | 0.288 / 238.6T | 37.9 / 1.8T | 132x |

The kernel is varlen-form (see Step 1) so the column measures the single kernel
with varlen indexing + the per-batch guard; perf is latency-bound (serial load
→ MMA → softmax per KV block, zero overlap), so shape only affects grid size.

### Key takeaways
- **Correctness is complete** for ex.1 varlen (8 shapes × varied lengths incl.
  misaligned + GQA, and 5 test_varlen cases) — all PASS at atol=0.016 vs torch
  SDPA, plus 3-way agreement with hpc-ops via compare_hpcops.py.
- **Performance is 1% of peak (vs hpc-ops 12-162%)** — fully latency-bound: the kernel is
  serial per KV block (load → sync → QK MMA → wait(0) → softmax → PV MMA → wait(0) → sync),
  zero overlap. Optimization steps below target this.

---

## Step 1: Baseline — varlen API + autovec_copy + single warpgroup

**Tag**: `flash-ex1-v1-varlen` (pending tag)

### Kernel design (ex.1 `prefill_bf16_multistage.py`, single class)
- hpc-ops compatible varlen API: `(total_seq, H_q/H_kv, D)` Q/K/V + `seqlens` +
  `cu_seqlens`, always causal, internal 1/sqrt(D) scale.
- Grid `(B*H_q, ceil(max_seqlens/64), 1)`, 128 threads (1 warpgroup), single-pass
  online softmax, per-batch guard (`q_tile_start < q_len`).
- QK WGMMA `(64, 64, 16)` atom `(1,1,1)` — N=64 embedded in instruction shape so all KV
  cols land in ONE warpgroup; 4 threads share an M row → `warp_reduction` covers the row
  (`reduction_target_n == (4,)`). (Using `(64,16,16)+(1,4,1)` needs 512 threads and
  computes only 1/4 of N — the early ~16x softmax bug.)
- PV WGMMA RS-scope: P from registers (`make_acc_into_op`), V from SMEM K-major
  (V pre-transposed + per-batch padded to `(B, H_kv, D, max_seqlens)` in the harness).
- Batch-aware causal mask via identity tensor (max_s × max_s) routed through
  `partition_C` + `layout_acc_mn`; rows past `q_len` and `k_col > q_row → -inf`.
- Serial loads: `autovec_copy` for Q/K/V each KV block + `sync_threads`.
- 64-aligned batch boundaries: `cu_seqlens[b]` must be a BLK_M multiple (kernel
  tiles gmem at 64 rows); harness pads each batch's flatten via `pack_varlen`.

### Performance (causal; hpc-ops column from compare_hpcops.py)

| Shape | ex.1 ms | TFLOPS | % peak | hpc-ops ms | Speedup |
|---|---|---|---|---|---|
| 512² | 1.108 | 0.5 | 0.3% | 0.030 | 37x |
| 1024² | 2.844 | 1.5 | 1.0% | 0.053 | 54x |
| 2048² | 11.49 | 1.5 | 1.0% | 0.095 | 121x |
| 4096² GQA | 37.6 | 1.8 | 1.2% | 0.286 | 132x |

Occupancy is 100% (16 blocks/SM × 128 threads) yet 1% peak → latency-bound, not
occupancy-bound. Root cause: per-block full serialization of load/compute with zero
overlap across KV blocks.

### Strengths
- Correct across GQA / causal / multi-batch / varlen (equal, unequal, misaligned seqlens).
- Teaching-clean structure (SS-scope QK, RS-scope PV, layout_acc_mn softmax).

### Bottlenecks (ncu hypothesis, to confirm in v2)
- Stalls on `sync_threads` + `wait_group(0)` after every load/MMA — no pipelining.
- TMA not used — `autovec_copy` blocks the issuing warp on the copy.
- No double buffering: K/V for block j+1 cannot load while block j computes.

---

## Step 2+: planned optimizations

| Step | Tag | Change | Expected |
|---|---|---|---|
| 2 | flash-ex1-v2-tma | TMA async loads + mbarrier; replace autovec_copy | hide gmem latency |
| 3 | flash-ex1-v3-doublebuf | NUM_STAGES=2 manual double-buffering (multi-stage, NOT warp-spec — that's ex.2) | overlap load/compute |
| 4 | flash-ex1-v4-causal-exit | skip future KV blocks (upper triangle) | 2x at M≈N causal |
| 5 | flash-ex1-v5-splitk | split-K over KV (grid z) + LSE combine | multi-CTA parallelism |

Each step: implement → verify (`run_prefill.py --ex=1` + `tests/test_varlen.py`
+ `compare_hpcops.py`) → record in Master Table → `git commit` + tag → ncu report if >10% jump.

---

## Other exercises (scaffolds, not yet verified)

> Note: the dense-form prefill scaffolds (ex.2 warpspec, ex.4 fp8) were deleted
> when the project went varlen-only — prefill keeps just ex.1 varlen.

| Exercise | File | Status | Notes |
|---|---|---|---|
| 3 | decode_bf16_splitk.py | 🚧 | decode + paged KV + split-K + LSE combine. Class-based, identity-tensor causal, RS PV (64,128,16). Not compile-tested after rewrite. |
| 5 | decode_fp8.py | 🚧 | FP8 decode, QK=SS/SV=RS. Not yet wired. |

Once ex.1 hits a stable optimized tag (v4+), the TMA/double-buffer pattern is ported to
each scaffold before profiling it. Per-exercise tables land here as each becomes verified.