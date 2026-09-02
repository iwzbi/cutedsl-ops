# FlashAttention Prefill (multi-stage) — Optimization Journey

> Per-kernel optimization log for `kernels/prefill_bf16_multistage.py` (ex.1).
> Scope: the single-warpgroup TMA multi-stage kernel only — warp-specialization
> (ex.2) is a different kernel and intentionally out of scope here.

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

| Shape (H_q,H_kv,D,seqlens) | hpc-ops | ex.1 v2-TMA | ex.1 v3 | vs hpc (v3) |
|---|---|---|---|---|
| **(4,4,128,[512])** single batch | 0.030 ms / 18.1T (12%) | 0.070 ms / 7.6T (5%) | **0.071 / 7.5T** | 2.2x slower |
| **(8,8,128,[1024])** single batch | 0.053 / 80.6T (55%) | 0.128 / 33.5T (23%) | **0.130 / 33.1T** | 2.4x slower |
| **(4,1,128,[512])** GQA | 0.029 / 18.2T | 0.070 / 7.7T | **0.071 / 7.5T** | 2.4x slower |
| **(1,1,128,[512])** single head | 0.029 / 4.7T | 0.072 / 1.9T | **0.073 / 1.8T** | 2.5x slower |
| **(4,4,128,[4096])** long seq † | 0.165 / 208.0T (>peak) | 0.350 / 98.1T (66%) | — † | 2.1x (v2) |
| **(8,2,128,[4096])** GQA long † | 0.288 / 238.6T (>peak) | 0.593 / 115.8T (78%) | — † | 2.1x (v2) |
| **(4,4,128,[512,768])** unequal | 0.044 / 39.5T | 0.120 / 14.6T | **0.120 / 14.5T** | 2.4x slower |
| **(4,4,128,[200,328])** misaligned | 0.026 / 11.7T | 0.066 / 4.6T | **0.068 / 4.4T** | 2.6x slower |
| **(4,1,128,[256,384,512])** GQA×3 | 0.035 / 28.0T | 0.110 / 8.8T | **0.110 / 8.9T** | 3.2x slower |
| **(4,1,128,[512]×4)** GQA×4 | 0.036 / 60.3T | 0.109 / 19.7T | **0.110 / 19.5T** | 3.1x slower |
| **(4,4,128,[2048,2048])** † | 0.090 / 190.2T | 0.263 / 65.4T (44%) | — † | 2.9x (v2) |
| **(4,4,128,[512]×8)** serving † | 0.040 / 107.8T | 0.196 / 21.9T | — † | 4.9x (v2) |
| **(8,8,128,[1024]×8)** serving † | 0.164 / 209.8T | 0.775 / 44.3T | — † | 4.7x (v2) |
| **(4,4,128,[512]×16)** serving † | 0.059 / 145.4T | 0.364 / 23.6T | — † | 6.2x (v2) |

† hpc-ops dispatches these (>156 CTAs on H20) to its **warp_spec** kernel
(`prefill.cc`: `ceil(max_seq/64)*B*H_q < 2·SM` → multi_stage, else warp_spec), so
comparing our single-WG kernel against them is structural mismatch — the v3+
bench target is the 8 multi-stage shapes only.

### Key takeaways
- **Correctness is complete** for ex.1 varlen: 14 PREFILL_SHAPES (single/multi-batch,
  equal/unequal/misaligned, GQA 1..16 groups, 64→4096 seqs) via `run_prefill.py`,
  5 test_varlen cases, and 14×3-way agreement (torch/hpc-ops/cutedsl) via compare_hpcops.py.
- **v1→v2 (TMA multi-stage): 16-100x faster** — the serial-load bottleneck is gone;
  the gap to hpc-ops collapsed from 37-132x to **2.1-6.2x**.
- **v3 ruled out the easy suspects** (see Step 3): occupancy (2→3 CTA/SM), L2 grid
  order, and per-tile mask skipping are all *neutral* — v2 already saturates what
  the single-warpgroup design can do *at fixed grid shape*.
- **v3 ncu attribution (Step 3.5)**: the residual gap on small shapes is
  *parallelism starvation* (32-CTA grid = 0.14 wave, tensor pipe 0.55%), not
  codegen quality. Fix = split-K over the KV dimension (Step 4).

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

Multi-batch shapes drift further behind (up to 6.2x on [512]×16) — these have
enough CTAs to fill the GPU, and the per-warp issue serialization plus the
larger KV working set expose the remaining single-warpgroup cost (see Step 3.5
for the ncu attribution).

### Verified
- `run_prefill.py`: 14/14 succeed (max diff ≤ 0.00195).
- `tests/test_varlen.py`: 5/5 passed.
- `compare_hpcops.py`: 14 shapes × 3-way (torch/hpc-ops/cutedsl) all Success.
- `make quality`: clean.

---

## Step 3: ⚖️ v3 — occupancy / L2 / mask-skip A/B study (neutral, kept for fidelity)

**Tag**: `flash-ex1-v3-occupancy`

### Changes (each mirrors an hpc-ops `multi_stage_dim128` trait; h20 dispatch:
`ceil(max_seq/64)·B·H_q < 156 CTAs` → the 8 small shapes below)

1. `NUM_STAGES 2 → 1`: smem 104KB→72KB → **3 CTAs/SM** (hpc runs kStage=1:
   48+16KB, 3 CTAs — latency hidden by CTA interleaving, not in-CTA prefetch).
2. **Grid transpose** `(B·H_q, q_tile) → (q_tile, B·H_q)`: Q-tile becomes the fast
   dim so one wave reads nested causal KV prefixes of the same head (L2 reuse),
   hpc's launch order.
3. **Full-tile mask skip** (`num_tile_full = bid_m`): KV tiles entirely under the
   diagonal ((itile+1)·64−1 < q_tile_start) skip the causal-mask loop — hpc does
   the same via `num_tile_full`.

### A/B result (8 multi-stage shapes, cute ms)

| Shape | v2 (s2) | v3a (s1+swap+skip) | v3b (s2+swap+skip) |
|---|---|---|---|
| (4,4,128,[512]) | 0.070 | 0.071 | 0.069 |
| (8,8,128,[1024]) | 0.128 | 0.130 | 0.129 |
| (4,1,128,[512]) | 0.070 | 0.071 | 0.069 |
| (1,1,128,[512]) | 0.072 | 0.073 | 0.070 |
| [512,768] | 0.120 | 0.120 | 0.119 |
| [200,328] | 0.066 | 0.068 | 0.066 |
| [256,384,512] | 0.110 | 0.110 | 0.108 |
| [512]×4 | 0.109 | 0.110 | 0.110 |

**All three changes are performance-neutral (≤4%)** — v2-TMA had already
saturated the single-warpgroup design: TMA + prefetch made loads invisible, so
occupancy (2→3 CTA), L2 order, and saved mask instructions cannot move the
needle. v3a config kept anyway (hpc-faithful: kStage=1 + grid order + mask skip;
mask-skip also shrinks the loop body executed on real workloads).

### What this rules out / next
- The residual 2.2-3.2x vs hpc-ops' *structurally identical* kernel is NOT about
  pipeline depth or occupancy in itself — Step 3.5's ncu run shows the true
  failure mode is an under-filled grid + exposed single-warp latency chain.

### Verified
- `run_prefill.py`: 14/14 succeed; `tests/test_varlen.py`: 5/5;
  `compare_hpcops.py --shapes 0,1,2,3,6,7,8,9`: 8×3-way all Success.

---

## Step 3.5: 🔬 ncu stall attribution (v3 config, shape (4,4,128,[512]))

Evidence: [`ncu_reports/ex1-v3-multistage-512.txt`](./ncu_reports/ex1-v3-multistage-512.txt)
(`ncu --set full`, kernel `kernel_cutlass_kernel_opsflash_attn...`, capture via
`NCU_PROFILING=1` + `--kernel-name regex:kernel_cutlass`).

| Signal | Value | Reading |
|---|---|---|
| Grid Size / Waves Per SM | 32 CTA / **0.14 wave** | 78-SM GPU < half filled; ≥1 CTA per SM never happens |
| Active Warps Per Scheduler | **1.00** (of 16) | one resident warp → zero warp-level interleave |
| Compute (SM) / DRAM throughput | 14.9% / **0.45%** | neither ALU nor bandwidth saturated |
| **Tensor pipe utilization** | **0.55%** | WGMMA idles ~99.5% of elapsed cycles |
| Stall Long Scoreboard | **1.57 cyc/inst (34%)** | exposed global-load latency (synchronous `autovec_copy` Q load + small-work CTAs) |
| Stall Wait | 1.09 (24%) | fixed-latency QK→softmax→PV dependency chain |
| Stall Short Scoreboard / Barrier | 0.33 / 0.22 | LDS + sync minor |

**Verdict: parallelism starvation, not codegen quality.** With 32 CTAs and no
in-CTA pipelining slack, every warp serializes Q-load → TMA-wait → WGMMA →
softmax with no co-resident work to cover latency; the tensor pipe never spins up.
hpc-ops' multi_stage kernel fights this with 64KB smem (3 CTA/SM) — still capped
by grid=32 on these shapes, which is why even *it* only reaches 12% peak at 512².

**Implication → Step 4 (split-K)**: splitting the KV range across `split ∈ {2,4}`
CTAs (grid z) + LSE combine multiplies the CTA count exactly where shapes are
small (32→64/128 CTAs fills the GPU; 512², [512]×4, [512,768], GQA×3 all gain),
and shrinks each CTA's serial KV-tile chain. Secondary lever from the long-scoreboard
finding: make the Q load TMA/async so tile-0's QK can overlap Q-in-flight.

---

## Step 4+: planned optimizations (multi-stage scope only)

| Step | Tag | Change | Expected |
|---|---|---|---|
| 4 | flash-ex1-v4-splitk | split-KV across grid z (2/4) + LSE combine; grid fills the 78-SM machine on small shapes | big win on the 8 multi-stage shapes (tensor pipe 0.55% → utilization by breadth) |
| 5 | flash-ex1-v5-qasync | Q load via TMA (or cp.async pipeline) so QK tile-0 doesn't pay the synchronous LDG chain | kills the 34% long-scoreboard head-on for short CTAs |
| 6 | (later) | smem aliasing (sO over sQ like hpc's 64KB budget) to lift occupancy at long seqs | polish |

Each step: implement → verify (`run_prefill.py` + `tests/test_varlen.py`
+ `compare_hpcops.py`) → record in Master Table → `git commit` + tag → ncu report if >10% jump.
**Bench target set from v3 on: the 8 multi-stage shapes** (`--shapes 0,1,2,3,6,7,8,9`) —
long/serving shapes compare against hpc-ops' *different* kernel (warp_spec) and are
informational only.

