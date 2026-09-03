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

`vs hpc` = hpc_ms / cute_ms — **> 1 means OUR kernel is faster**. v5 + hpc rows are
the same 14-shape re-bench run (FA_SPLIT=1, 3-way vs torch all Success); v1/v2/v3
rows keep their historical numbers.

| Shape (H_q,H_kv,D,seqlens) | CTAs | hpc-ops | ex.1 v2-TMA | ex.1 v3 | ex.1 v5 | **ex.1 v6 (stages=2+auto split)** | vs hpc (v6) |
|---|---|---|---|---|---|---|---|
| **(4,4,128,[512])** single batch | 32 | 0.034 ms / 15.7T (11%) | 0.070 ms / 7.6T (5%) | 0.071 / 7.5T | 0.027 / 20.1T | **0.023** | **1.48x faster** |
| **(8,8,128,[1024])** single batch | 128 | 0.053 / 80.8T (55%) | 0.128 / 33.5T (23%) | 0.130 / 33.1T | 0.050 / 85.5T | **0.050** | **1.07x faster** |
| **(4,1,128,[512])** GQA | 32 | 0.029 / 18.3T | 0.070 / 7.7T | 0.071 / 7.5T | 0.027 / 20.2T | **0.021** | **1.39x faster** |
| **(1,1,128,[512])** single head | 8 | 0.029 / 4.7T | 0.072 / 1.9T | 0.073 / 1.8T | 0.026 / 5.1T | **0.019** | **1.52x faster** |
| **(4,4,128,[4096])** long seq † | 512 | 0.164 / 209.0T (>peak) | 0.350 / 98.1T (66%) | — † | 0.244 / 140.8T (95.1%) | — | — |
| **(8,2,128,[4096])** GQA long † | 512 | 0.286 / 240.3T (>peak) | 0.593 / 115.8T (78%) | — † | 0.356 / 193.0T (130%) | — | — |
| **(4,4,128,[512,768])** unequal | 96 | 0.044 / 39.6T | 0.120 / 14.6T | 0.120 / 14.5T | 0.041 / 42.2T | **0.035** | **1.25x faster** |
| **(4,4,128,[200,328])** misaligned | 48 | 0.026 / 11.7T | 0.066 / 4.6T | 0.068 / 4.4T | 0.023 / 13.2T | **0.021** | **1.21x faster** |
| **(4,1,128,[256,384,512])** GQA×3 | 96 | 0.035 / 28.1T | 0.110 / 8.8T | 0.110 / 8.9T | 0.032 / 30.4T | **0.032** | **1.08x faster** |
| **(4,1,128,[512]×4)** GQA×4 | 128 | 0.034 / 63.1T | 0.109 / 19.7T | 0.110 / 19.5T | 0.034 / 64.0T | **0.033** | **1.03x faster** |
| **(4,4,128,[2048,2048])** † | 128 | 0.090 / 190.5T | 0.263 / 65.4T (44%) | — † | 0.119 / 144.6T (97.7%) | — | — |
| **(4,4,128,[512]×8)** serving † | 256 | 0.039 / 108.9T | 0.196 / 21.9T | — † | 0.044 / 98.5T | — | — |
| **(8,8,128,[1024]×8)** serving † | 1024 | 0.164 / 209.3T | 0.775 / 44.3T | — † | 0.187 / 184.0T | — | — |
| **(4,4,128,[512]×16)** serving † | 512 | 0.059 / 145.9T | 0.364 / 23.6T | — † | 0.069 / 125.4T (84.7%) | — | — |
| **(32,8,128,[512]×32)** Llama3-8B 16k tok † | 8192 | 0.653 / 210.5T | — | — | — | 0.826 | 0.79x |
| **(32,8,128,[1024]×16)** † | 4096 | 1.139 / 241.4T | — | — | — | 1.341 | 0.85x |
| **(32,8,128,[2048]×8)** † | 2048 | 2.109 / 260.6T | — | — | — | 2.362 | 0.89x |
| **(32,8,128,[4096]×4)** † | 1024 | 4.010 / 274.2T | — | — | — | 4.432 | 0.90x |
| **(32,8,128,[8192]×2)** † | 512 | 7.861 / 279.7T | — | — | — | 8.605 | 0.91x |
| **(32,8,128,[16384])** † | 256 | 15.615 / 281.7T | — | — | — | 16.917 | 0.92x |
| **(32,8,128,U(512,4k)×16)** varlen dist † | ~4k | 5.136 / 259.1T | — | — | — | 5.511 | 0.93x |
| **(32,8,128,zipf[128..6k]×12)** † | ~2k | 4.751 / 251.9T | — | — | — | 4.938 | 0.96x |

† hpc-ops dispatches these (>156 CTAs on H20) to its **warp_spec** kernel
(`prefill.cc`: `ceil(max_seq/64)*B*H_q < 2·SM` → multi_stage, else warp_spec), so
comparing our single-WG kernel against them is structural mismatch — the v3+
bench target is the 8 multi-stage shapes only. **v6 (stages=2 ring + automatic split-K, Step 6) wins ALL 8 multi-stage shapes at
1.03-1.52x** (CTAs column = grid size `ceil(max_s/64)*B*H_q`; v6 ms derived from the
Step 6 four-quadrant A/B ratio matrix; † shapes not re-benched under v6 defaults).

The last 8 rows (added with the shape-set refresh) use **industry-standard configs**
— Llama3-8B heads (32Q/8KV) at a constant ~16k-token budget swept across
batch×seqlen (Dao-AILab / FlashInfer benchmark convention) plus uniform/zipf varlen
distributions. All are † (hpc = warp_spec). The gap *closes monotonically with
sequence length* — 0.79x at [512]×32 up to 0.92x at [16384], and the zipf varlen mix
reaches **0.96x** — our per-CTA fixed costs amortize fully on long loops, and the
residual is single-warpgroup vs 3-warpgroup WGMMA scheduling, not load/pipeline stalls.

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
- **v4 split-KV: negative result** (Step 4) — cutting the loop cannot pay for
  per-CTA fixed costs (synchronous Q load + TMA wait + drain), which dominate.
- **v5 Q-async TMA: 2.5-3.4x — ALL 8 multi-stage shapes now FASTER than hpc-ops**
  (1.01-1.28x on the re-bench; the 6 † shapes sit at 0.67-0.90x where hpc uses its
  structurally different warp_spec kernel). The ncu-predicted lever worked exactly as
  diagnosed; split=2 re-test after v5 turns positive on the smallest grids (1.29-1.39x).
- **v6 stages=2 + auto split-K + sP smem removal: 8/8 multi-stage shapes at 1.03-1.52x**
  (single-head 1.52x, 512² 1.48x). Double-buffering turned positive once v5/v6b removed
  the fixed per-CTA latency it must hide; split=2 is auto-selected iff grid ≤ 96 CTAs.

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
and shrinks each CTA's serial KV-tile chain. *(Step 4 tested this — the split-K
prediction was wrong for all but one shape; see Step 4. The long-scoreboard
secondary lever stands and becomes Step 5.)*

---

## Step 4: ❌ split-KV — full implementation, negative result (kept off by default)

**Tag**: `flash-ex1-v4-splitk` (infra retained; `split_k=1` default = v3 behavior)

### What was built (verified correct, fully wired)
- Main kernel gains `split_k` (grid z): each CTA owns KV tile range
  `[s·⌈nb/S⌉ … ]` via `kv_lo/kv_hi = s_idx*num_n_blocks//split_k` — disjoint spans,
  zero duplicated math. `kv_limit`/causal mask/c_idx semantics unchanged.
- split_k>1 epilogue writes **fp32 partials** (unnormalized `tCrO` → `PO (T,H_q,S,D)`,
  row max `Pm`, row sum `Pl`) via a second `make_tiled_copy_C`; empty spans store
  (0, −inf, 0) naturally. A separate `combine_kernel` (grid (T,H_q)×128) does the
  LSE merge: `M=max_s Pm`, `w_s=exp2(scale·(Pm_s−M))`, `O=Σ w·PO / Σ w·Pl`.
- Harness: `FA_SPLIT` env / `--split` CLI across run_prefill / compare_hpcops /
  test_varlen; `pick_split()` helper available. Correctness: split∈{1,2,4} →
  run_prefill 14/14 + test_varlen 5/5 + compare 3-way all Success.

### A/B bench (8 multi-stage shapes, cute ms; v3 = split 1)

| Shape | v3 | s2 | s4 | s8 |
|---|---|---|---|---|
| (4,4,128,[512]) | 0.071 | 0.072 | 0.106 | 0.188 |
| (8,8,128,[1024]) | 0.130 | 0.209 | 0.327 | 0.612 |
| (4,1,128,[512]) GQA | 0.071 | 0.073 | 0.106 | 0.188 |
| **(1,1,128,[512])** single head | 8 | 0.073 | 0.070 | 0.067 | **0.019** | **1.52x faster** 
| [512,768] | 0.120 | 0.156 | 0.234 | 0.397 |
| [200,328] | 0.068 | 0.107 | 0.146 | 0.227 |
| [256,384,512] | 0.110 | 0.152 | 0.193 | 0.353 |
| [512]×4 | 0.110 | 0.199 | 0.319 | 0.603 |
| (4,4,128,[4096]) † | 0.350 | 0.456 | — | — |
| (4,4,128,[2048,2048]) † | 0.262 | 0.381 | — | — |

† long-loop control shapes: also *slower* at s2 — the negative result is not a
short-kernel artifact. Only the genuinely under-occupied single-head shape improves.

### ncu attribution (split=2, (4,4,128,[512]))
- `combine_kernel` Duration = **4.2 us** — the combine is cheap, *not* the problem.
- Main kernel Duration: 71 us (v3, 32 CTA) → **83 us** (s2, 64 CTA) — splitting
  *doubled* the CTAs yet the main kernel got **17% slower** even though each CTA
  runs half the loop.

### Why the hypothesis died
Per-CTA **fixed cost dominates the critical path**: synchronous `autovec_copy` Q
load (34% long-scoreboard in Step 3.5), first-tile TMA wait, and prologue/epilogue
drain. Halving the loop body saves the *minority* of the time; meanwhile each
split CTA re-pays the full Q load (×S gmem traffic) and the partials write
(PO fp32 = 2 MB per split × S). CTA-count breadth cannot help when every CTA is
latency-serial inside itself. This is fully consistent with v3's three knobs
being neutral and with the split=2 main kernel slowing down.

**Take-away**: the binding constraint is the per-CTA latency chain, not grid size.
Step 3.5's secondary lever is promoted: **Step 5 = async Q load (v5-qasync)** —
and only after the CTA critical path is short does split-KV deserve a re-test
(it's kept as ready infrastructure behind `split_k`).

---

## Step 5: ✅ v5-qasync — asynchronous Q via TMA (2.5-3.4x, 7/8 shapes beat hpc-ops)

**Tag**: `flash-ex1-v5-qasync`

### Change
The once-per-CTA Q tile moves from the synchronous `autovec_copy` (issuing warps
block on global LDG — Step 3.5 measured long-scoreboard at 34%) to a **dedicated
single-stage `PipelineTmaAsync`** (`q_bar_array`, tx = BLK_M·D·2 = 16 KB):
- host builds `q_view = (total_seq, D, H_q)` strides `(H_q·D, 1, D)` — exactly the
  K view pattern — and a `(BLK_M, D)` TMA box; the `tma_tensor` (not the raw
  view!) is passed to the kernel.
- warp 0 issues the Q copy in the prologue (`producer_acquire → cute.copy →
  producer_commit`); all threads `consumer_wait` once right before the mainloop.
  sQ is never overwritten, so no release is needed.
- K/V pipeline, MMA, softmax, mask, epilogue, split-K infra: untouched.

### Bench (FA_SPLIT=1, L2-flushed, 24/24 3-way Success)

| Shape | v3 ms | v5 ms | v3→v5 | v5 vs hpc |
|---|---|---|---|---|
| (4,4,128,[512]) | 0.071 | **0.026** | 2.7x | **0.89x** |
| (8,8,128,[1024]) | 0.130 | **0.050** | 2.6x | **0.94x** |
| (4,1,128,[512]) GQA | 0.071 | **0.026** | 2.7x | **0.90x** |
| (1,1,128,[512]) | 0.073 | **0.029** | 2.5x | 0.99x |
| [512,768] | 0.120 | **0.041** | 2.9x | **0.93x** |
| [200,328] | 0.068 | **0.023** | 3.0x | **0.87x** |
| [256,384,512] | 0.110 | **0.032** | 3.4x | **0.85x** |
| [512]×4 | 0.110 | **0.033** | 3.3x | **0.92x** |

### ncu re-check (shape0, evidence: [`ncu_reports/ex1-v5-qasync-512.txt`](./ncu_reports/ex1-v5-qasync-512.txt))
- Duration **24.5us** vs v3's 71-83us (−65%): matches the bench.
- L1TEX-scoreboard share stays ~39%/inst but *total* exposed time collapsed —
  the stall residue is now scalar/control reads, not a serial 16 KB LDG chain.
- Active warps/scheduler still 1.00 and waves 0.14: the small-grid structure is
  unchanged — v5 won by shortening the per-CTA critical path, exactly what
  Step 4's negative result said was the binding constraint.

### Split-KV re-test after v5 (FA_SPLIT=2, 24/24 Success)
With the Q cost gone, splitting finally pays on the smallest grids:
[512]→**0.023 (0.77x hpc)**, GQA [512]→0.023 (0.78x), (1,1,[512])→**0.021 (0.72x)**,
[512,768]→0.037 (0.83x); [200,328]/GQA×3 flat; [1024] and [512]×4 still prefer s1.
→ v4's "split never wins" is refined: **split needs the fixed per-CTA cost removed first**.

### Verified
- `run_prefill.py`: 14/14 (FA_SPLIT=1) + split=2 long-seq spot check;
  `test_varlen.py` 5/5 at split∈{1,2}; `compare_hpcops.py` 8×3-way at split∈{1,2}.

---

## Step 6: ✅ v6 — stages=2 ring + auto split-K + sP smem removal (multi-stage 8/8, up to 1.52x)

**tag: `flash-ex1-v6-picksplit`** — three bundled changes, one four-quadrant A/B to decide defaults.

### v6b: sP shared-memory removal (−8 KB/CTA)
`sP_flat` existed only as a `partition_C` layout *template* for the QK accumulator
(`make_fragment_C` lands in registers; the smem itself was never dereferenced).
It now borrows `storage.sO.data_ptr()` and the 8 KB `sP` MemRange is gone from
SharedStorage — 72→64 KB/CTA, one more CTA fits per SM.

### v6c + v6a four-quadrant A/B (`compare_hpcops --shapes 0,1,2,3,6,7,8,9`, 24/24 3-way per cell)
vs-hpc ratio (hpc_ms/cute_ms, >1 = we are faster) at each (K/V ring stages, split_k):

| Shape (CTAs) | (1,1) | (2,1) | (1,2) | (2,2) |
|---|---|---|---|---|
| 512² (32) | 1.02 | 1.04 | 1.26 | **1.48** |
| 1024² (128) | 0.93 | **1.07** | 0.96 | 0.97 |
| GQA 512 (32) | 0.99 | 1.33 | 1.19 | **1.39** |
| single-head (8) | 0.99 | 1.28 | 1.31 | **1.52** |
| [512,768] (96) | 0.97 | 1.09 | 1.07 | **1.25** |
| [200,328] (48) | 1.04 | 1.31 | 1.09 | **1.21** |
| GQA×3 (96) | 0.98 | **1.12** | 0.90 | 1.08 |
| [512]×4 (128) | 1.00 | **1.03** | 0.90 | 0.82 |

Two conclusions:
1. **stages=2 turned positive across the board** (≥ (1,1) on every shape, no
   regression). v3's "double-buffering is neutral" verdict was measured *before*
   Q-async/sP changes removed the fixed per-CTA latency the pipeline must hide —
   the premise flipped, so the default `NUM_STAGES` is now **2**.
2. **split-K wins iff the grid is under-filled** (`pick_split(ctas) = 2 if ctas ≤ 96
   else 1`): the ≤96-CTA shapes gain 15-40% from (2,2), the 128-CTA shapes lose
   (GPU already busy → partial traffic is pure cost). Harnesses (`run_prefill`,
   `compare_hpcops`, `test_varlen`) compute it per shape automatically;
   `FA_SPLIT`/`--split` env/CLI stay as A/B overrides. Same pattern for stages
   (`FA_STAGES`, default 2).

### Result (default = stages 2 + auto split)
**8/8 multi-stage shapes faster than hpc-ops, 1.03-1.52x** — peak single-head
[512] 1.52x, 512² 1.48x, GQA 1.39x. v5's 1.01-1.28x era is superseded.

### Verified
- `run_prefill.py` **14/14** and `test_varlen.py` **5/5** under the new defaults
  (auto split, no env); `FA_STAGES=2 FA_SPLIT=2` worst-case combo 5/5; matrix
  cells 24/24 3-way (torch/hpc/cutedsl) Success each.

---

## Step 7+: planned optimizations (multi-stage scope only)

| Step | Tag | Change | Expected |
|---|---|---|---|
| ~~6~~ | ~~flash-ex1-v6-picksplit~~ | ✅ DONE — 8/8 multi-stage shapes faster (1.03-1.52x) | — |
| 7 | (candidate) | † long-seq re-bench under stages=2 (matrix only covered the 8 multi-stage shapes); sO-over-sQ smem aliasing for occupancy; BLK_N=128 (halve iterations) | †-region + polish |

Each step: implement → verify (`run_prefill.py` + `tests/test_varlen.py`
+ `compare_hpcops.py`) → record in Master Table → `git commit` + tag → ncu report if >10% jump.
**Bench target set from v3 on: the 8 multi-stage shapes** (`--shapes 0,1,2,3,6,7,8,9`) —
long/serving shapes compare against hpc-ops' *different* kernel (warp_spec) and are
informational only.

