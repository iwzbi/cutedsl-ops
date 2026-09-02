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
(our kernel + hpc-ops via `compare_hpcops.py`). TFLOPS = 4·H_q·Σs²·D / t.
NOTE: both kernels skip the causal upper triangle via `kv_limit`, so this formula
overcounts their work equally at M≈N (hpc-ops reports >100% peak for exactly this
reason) — the hpc/cute *ratio* is fair, the absolute TFLOPS is an upper bound.

| Shape (H_q,H_kv,D,seqlens) | hpc-ops | ex.1 v2-TMA | v2 vs hpc |
|---|---|---|---|
| **(4,4,128,[512])** single batch | 0.030 ms / 18.1T (12%) | **0.070 ms / 7.6T (5%)** | 2.4x slower |
| **(8,8,128,[1024])** single batch | 0.053 / 80.6T (55%) | **0.128 / 33.5T (23%)** | 2.4x slower |
| **(4,1,128,[512])** GQA | 0.029 / 18.2T | **0.070 / 7.7T** | 2.4x slower |
| **(1,1,128,[512])** single head | 0.029 / 4.7T | **0.072 / 1.9T** | 2.5x slower |
| **(4,4,128,[4096])** long seq | 0.165 / 208.0T (>peak) | **0.350 / 98.1T (66%)** | 2.1x slower |
| **(8,2,128,[4096])** GQA long | 0.288 / 238.6T (>peak) | **0.593 / 115.8T (78%)** | **2.1x slower** |
| **(4,4,128,[512,768])** unequal | 0.044 / 39.5T | **0.120 / 14.6T** | 2.7x slower |
| **(4,4,128,[200,328])** misaligned | 0.026 / 11.7T | **0.066 / 4.6T** | 2.6x slower |
| **(4,1,128,[256,384,512])** GQA×3 | 0.035 / 28.0T | **0.110 / 8.8T** | 3.2x slower |
| **(4,1,128,[512]×4)** GQA×4 | 0.036 / 60.3T | **0.109 / 19.7T** | 3.1x slower |
| **(4,4,128,[2048,2048])** | 0.090 / 190.2T | **0.263 / 65.4T (44%)** | 2.9x slower |
| **(4,4,128,[512]×8)** serving | 0.040 / 107.8T | **0.196 / 21.9T** | 4.9x slower |
| **(8,8,128,[1024]×8)** serving | 0.164 / 209.8T | **0.775 / 44.3T** | 4.7x slower |
| **(4,4,128,[512]×16)** serving | 0.059 / 145.4T | **0.364 / 23.6T** | 6.2x slower |

### Key takeaways
- **Correctness is complete** for ex.1 varlen: 14 PREFILL_SHAPES (single/multi-batch,
  equal/unequal/misaligned, GQA 1..16 groups, 64→4096 seqs) via `run_prefill.py`,
  5 test_varlen cases, and 14×3-way agreement (torch/hpc-ops/cutedsl) via compare_hpcops.py.
- **v1→v2 (TMA multi-stage): 16-100x faster** — the serial-load bottleneck is gone;
  the gap to hpc-ops collapsed from 37-132x to **2.1-6.2x**.
- Remaining gap is *not* causal skipping (both kernels already cap KV via `kv_limit`):
  it is single-warpgroup issue width + softmax/MMA serialization (→ ex.2 warp-spec)
  and small-grid / bandwidth effects on long sequences (→ num_stages, split-K tuning).

---

## Step 1: Baseline — varlen API + autovec_copy + single warpgroup

**Tag**: `flash-ex1-v1-varlen`

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

### Bottlenecks
- Stalls on `sync_threads` + `wait_group(0)` after every load/MMA — no pipelining.
- TMA not used — `autovec_copy` blocks the issuing warp on the copy.
- No double buffering: K/V for block j+1 cannot load while block j computes.

---

## Step 2: ✅ v2-TMA — TMA async loads + multi-stage ring (16-100x faster)

**Tag**: `flash-ex1-v2-tma`

### Kernel design (single warpgroup + PipelineTmaAsync, mirrors hpc-ops `multi_stage_dim128`)
- K and V load via **TMA** (`make_tiled_tma_atom` + `tma_partition`, 2D tile boxes
  `(BLK_N, Dd)` / `(Dd, BLK_N)`) driven by `pipeline.PipelineTmaAsync` with
  **`num_stages=2` ring buffers** (sK/sV get a stage mode; SharedStorage carries a
  `bar_kv_array` of 2·num_stages Int64 mbarriers). K and V share one pipeline
  (`tx_count` = K bytes + V bytes), so a single `consumer_wait` gates both.
- Producer = warp 0 (`if warp_idx == 0`), consumer = all 128 threads; prologue
  prefetches `num_stages-1` tiles, mainloop prefetches tile `itile+(kStage-1)`
  while computing tile `itile` (ring index + phase handled by PipelineState).
- TMA atoms MUST consume the **`tma_tensor` returned by `make_tiled_tma_atom`**
  (the plain gmem view / `make_tensor(iterator, …)` gives a `g_stride = "()"`
  empty atom that fails IR verification at `cute.copy`).
- Q stays autovec-loaded (once per CTA, off the critical path); O epilogue stays
  r2s `cute.copy`. MMA/softmax/mask structure identical to Step 1.
- gmem views per hpc-ops: K as `(S, D, H_kv)` strides `(H*D, 1, D)`, V as
  `(D, S, B*H_kv)` strides `(S, 1, D*S)` (V is K-major from `pack_varlen`).

### Gotchas discovered (each cost a debug round)
1. **`consumer_group` arrive-count must equal the number of *signaling* threads**
   (1 per warp = `NUM_THREADS // 32` = 4), NOT all 128 threads — DSL marks only
   `tidx % 32 < cluster_size` threads as release-arrivers (`sm90.py`
   `init_empty_barrier_arrive_signal`). With count=128 the empty barrier never
   flips → producer deadlocks (hang/`cudaErrorLaunchFailure` 719).
2. **Never wrap a TMA `cute.copy` in `elect_one()`** — the partitioning already
   makes a single thread issue it; elect_one causes GPU deadlock (DSL doc warning).
3. **`producer_commit` is a no-op for TMA** — the `arrive_and_expect_tx` already
   happened in `producer_acquire`; the TMA engine completes the transaction count.
4. **V's per-batch S extent must be 64-padded in the harness** (`pack_varlen`
   `v_t = (B, H_kv, D, S_pad)`): TMA reads whole BLK_N-wide tiles, and a
   non-aligned `max(seqlens)` makes the last V tile read past the batch plane.
   All callers pass `v_t.shape[3]` as the kernel's `max_seqlens`. (Note:
   `shape[2]` is **D**, `shape[3]` is S — a one-index typo silently corrupted
   every shape.)

### Performance (causal; hpc-ops column from compare_hpcops.py)

| Shape | v1 ms | v2 ms | v2 speedup vs v1 | vs hpc-ops |
|---|---|---|---|---|
| 512² | 1.108 | 0.070 | **15.8x** | 2.4x slower |
| 1024² | 2.844 | 0.128 | **22.2x** | 2.4x slower |
| 4096² GQA | 37.6 | 0.593 | **63x** | 2.1x slower |
| 2048²×2 | 11.49 | 0.263 | **44x** | 2.9x slower |
| [512]×16 | — | 0.364 | — | 6.2x slower |

Multi-batch shapes drift further behind (up to 6.2x on [512]×16): more CTAs than
SMs, each CTA's single warpgroup issues both TMA prefetch and WGMMA/softmax
serially — the warp-specialized overlap (load-WG vs compute-WG) is exactly what
ex.2/hpc-ops add; see Step 3.

### Verified
- `run_prefill.py`: 14/14 succeed (max diff ≤ 0.00195).
- `tests/test_varlen.py`: 5/5 passed.
- `compare_hpcops.py`: 14 shapes × 3-way (torch/hpc-ops/cutedsl) all Success.
- `make quality`: clean.

---

## Step 3+: planned optimizations

| Step | Tag | Change | Expected |
|---|---|---|---|
| 3 | flash-ex1-v3-stages | tune `num_stages` (3/4) + smem budget per occupancy | tighter load/compute overlap |
| 4 | flash-ex1-v4-splitk | split-K over KV (grid z) + LSE combine | multi-CTA parallelism for long seq |
| 5 | (later) | Q/O TMA paths, causal full-tile mask-skip (skip mask work for tiles fully below the diagonal) | polish |

Each step: implement → verify (`run_prefill.py` + `tests/test_varlen.py`
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
