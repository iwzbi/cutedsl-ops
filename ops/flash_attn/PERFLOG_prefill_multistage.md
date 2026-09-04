# FlashAttention Prefill (multi-stage) — Optimization Journey

> **STATUS: FROZEN at v10** (`flash-ex1-v10-frozen` = the unified-re-bench
> anchor). One gain merged post-freeze: **v11 PDL** (`flash-ex1-v11-pdl`) is the
> shipped code terminal; **v12** (cluster+DSMEM single-kernel merge) was built,
> measured slower, and reverted (Step 12). All 9 frozen versions were re-measured
> on one idle H20 under one unified protocol (same harness, same 22 shapes,
> same-session hpc-ops baseline) + per-version ncu; the quantified
> effect/metric/mechanism of every lever is in the Master Table and each Step's
> `Δ analysis` block.
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

## Master Performance Table (9 frozen versions re-measured on one idle H20 + v11)

Unified protocol: every tagged kernel runs the CURRENT harness — same 22
PREFILL_SHAPES, same `pack_varlen`, same `cuda_bench` (L2-flushed CUDA events),
hpc-ops measured in the same session, GPU idle, each version at its shipped
defaults (v1-v5 stages=1; v6+ stages=2 + auto `pick_split`; v7+ vectorized
combine; v8+ TMA-store epilogue; v11 = v8+ with PDL on the combine launch).
v11 was benched post-freeze over the 8 multi-stage shapes only (its matrix
column is `—` elsewhere). BF16 in / FP32 acc / BF16 out, causal varlen.
`vs hpc` = hpc_ms/cute_ms — **> 1 means our kernel is faster** (old `speedup`
label inverted accordingly). TFLOPS = 4·H_q·Σs²·D/t is a shared-convention
upper bound (both kernels skip the causal upper triangle, hence hpc >100% peak
figures). Raw data: `.omo/closure/{tag}_unified.tsv` + `matrix.json`; ncu:
`ncu_reports/` + `.omo/closure/{tag}_ncu.txt`.

### Version journey (reference shape `(4,4,128,[512])`, ncu = fused shape0 replay, ~1.5x inflated, use for cross-version trend only)

| Ver (tag) | One-line change | cute ms | vs hpc | step Δ | ncu Duration | top stall (samples) | tensor% |
|---|---|---|---|---|---|---|---|
| v1 `flash-ex1-v1-varlen` | baseline: autovec gmem→reg loads, serial | 1.008 | 0.031x | — | 1490 µs | long_scoreboard 12131 + wait 5670 | 0.03% |
| v2 `…-v2-tma` | K/V via TMA + mbarrier multi-stage ring | 0.070 | 0.44x | **16x** | 86.3 µs | long_scoreboard 710 (was 12131) | 0.57% |
| v3 `…-v3-occupancy` | stages 2→1, grid swap, mask-skip | 0.071 | 0.44x | neutral | 89.5 µs | long_scoreboard 842 | 0.55% |
| v4 `…-v4-splitk` | split-KV infra (default OFF after A/B) | 0.071 | 0.42x | neutral-off | 87.4 µs | long_scoreboard 727 | 0.56% |
| v5 `…-v5-qasync` | **Q via async TMA pipeline** | 0.027 | 1.17x | **2.6x** | 24.4 µs | long_scoreboard 174, barrier 112 ↑ | 2.15% |
| v6 `…-v6-picksplit` | stages=2 default + auto split≤96CTA + sP smem removed | 0.023 | 1.28x | 1.3x | 19.1 µs | **barrier 113 (top)** | 2.67% |
| v7 `…-v7-combine` | vectorized LSE combine + sO smem removed | 0.023 | 1.48x | split-zone −4-7% | 19.2 µs | barrier 119 (top) | 2.63% |
| v8 `…-v8-tma-store` | O epilogue = bulk TMA store from sQ-aliased pad | 0.022 | 1.40x | fused mid-band −14% | 17.6 µs | barrier 118 (top), long_sb 54 ↓ | 2.88% |
| v10 `…-v10-store-retire` | drop redundant bulk commit/wait | 0.022 | 1.39x | neutral (simplification) | 17.8 µs | barrier 114 (top) | 2.90% |
| v11 `…-v11-pdl` | PDL on the split-path combine launch (`use_pdl` + `griddepcontrol_wait`) | 0.020 | 1.45-1.79x | split-band −4~14% | — (fused path untouched) | — | — |
| v12 (no tag, **reverted**) | single-kernel split via Hopper cluster + DSMEM merge (Step 12) | 0.020-0.042 | 0.96-1.50x | **+0~35% regression** | — | — | — |

### Full 22-shape × 9-version matrix (vs hpc, >1 = ours faster; unified re-bench)

† = hpc dispatches to its 3-warpgroup persistent **warp_spec** kernel (CTAs>156);
the 8 non-† rows are its multi-stage zone (same structure as ours).

v11 column: dual-run mean of the 8 multi-stage shapes only (PDL touches just the
split=2 combine launch; big-grid † rows not re-run, expected neutral). v10-vs-v11
cute_ms: 512² 0.022→0.020, GQA512 0.022→0.019, H1 0.020→0.018, [512,768]
0.033→0.031, GQA×3 0.026→0.025, [200,328] 0.020→0.021 (sole regression);
split=1 controls 1024²/[512]×4 within noise.

| # | Shape (H_q,H_kv,D,seqlens) | hpc ms | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v10 | **v11** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0  | (4,4,[512]) | 0.031 | 0.03 | 0.44 | 0.44 | 0.42 | 1.17 | 1.28 | 1.48 | 1.40 | 1.39 | **1.57** |
| 1  | (8,8,[1024]) | 0.053 | 0.02 | 0.42 | 0.42 | 0.42 | 1.07 | 1.05 | 1.07 | 1.10 | 1.12 | **1.09** |
| 2  | (4,1,[512]) GQA | 0.030 | 0.03 | 0.43 | 0.43 | 0.43 | 1.14 | 1.26 | 1.25 | 1.35 | 1.35 | **1.53** |
| 3  | (1,1,[512]) 1-head | 0.030 | 0.03 | 0.43 | 0.43 | 0.41 | 1.15 | 1.23 | 1.23 | 1.34 | 1.35 | **1.61** |
| 4† | (8,2,[4096]) | 0.286 | 0.01 | 0.49 | 0.53 | 0.52 | 0.82 | 0.78 | 0.79 | 0.80 | 0.80 | — |
| 5† | (4,4,[4096]) | 0.164 | 0.01 | 0.47 | 0.44 | 0.44 | 0.71 | 0.70 | 0.70 | 0.69 | 0.70 | — |
| 6  | (4,4,[512,768]) | 0.045 | 0.02 | 0.38 | 0.42 | 0.38 | 1.10 | 1.10 | 1.10 | 1.19 | 1.17 | **1.45** |
| 7  | (4,4,[200,328]) | 0.026 | 0.04 | 0.41 | 0.47 | 0.40 | 1.21 | 1.28 | 1.26 | 1.40 | 1.39 | **1.19** |
| 8  | (4,1,[256,384,512]) | 0.036 | 0.03 | 0.33 | 0.39 | 0.32 | 1.13 | 1.14 | 1.12 | 1.26 | 1.26 | **1.36** |
| 9  | (4,1,[512]×4) | 0.036 | 0.03 | 0.33 | 0.34 | 0.32 | 1.11 | 1.07 | 1.15 | 1.21 | 1.29 | **1.19** |
| 10† | (4,4,[2048,2048]) | 0.090 | 0.01 | 0.35 | 0.32 | 0.32 | 0.78 | 0.74 | 0.72 | 0.74 | 0.79 | — |
| 11† | (4,4,[512]×8) | 0.040 | 0.02 | 0.21 | 0.24 | 0.22 | 0.95 | 0.91 | 0.89 | 0.99 | 0.99 | — |
| 12† | (8,8,[1024]×8) | 0.163 | 0.01 | 0.21 | 0.23 | 0.23 | 0.90 | 0.85 | 0.84 | 0.90 | 0.91 | — |
| 13† | (4,4,[512]×16) | 0.059 | 0.02 | 0.17 | 0.18 | 0.18 | 0.85 | 0.84 | 0.95 | 0.96 | 0.97 | — |
| 14† | (32,8,[512]×32) Llama3 | 0.648 | 0.02 | 0.12 | 0.13 | 0.13 | 0.88 | 0.79 | 0.79 | 0.92 | 0.93 | — |
| 15† | (32,8,[1024]×16) | 1.132 | 0.01 | 0.20 | 0.22 | 0.22 | 0.94 | 0.85 | 0.85 | 0.91 | 0.92 | — |
| 16† | (32,8,[2048]×8) | 2.094 | 0.01 | 0.32 | 0.38 | 0.38 | 0.95 | 0.89 | 0.89 | 0.90 | 0.91 | — |
| 17† | (32,8,[4096]×4) | 4.034 | 0.01 | 0.46 | 0.58 | 0.58 | 0.96 | 0.91 | 0.90 | 0.89 | 0.90 | — |
| 18† | (32,8,[8192]×2) | 7.864 | 0.01 | 0.61 | 0.78 | 0.79 | 0.97 | 0.92 | 0.92 | 0.89 | 0.89 | — |
| 19† | (32,8,[16384]) | 15.568 | 0.01 | 0.74 | 0.91 | 0.91 | 0.97 | 0.92 | 0.93 | 0.89 | 0.89 | — |
| 20† | (32,8,U(512,4k)×16) | 5.137 | 0.01 | 0.39 | 0.46 | 0.46 | 1.00 | 0.94 | 0.93 | 0.94 | 0.95 | — |
| 21† | (32,8,zipf×12) | 4.758 | 0.02 | 0.47 | 0.57 | 0.57 | 1.03 | 0.96 | 0.97 | 0.96 | 0.97 | — |

### Final standing vs hpc-ops (v10)
- **multi-stage zone (8 shapes, same-structure opponent): 8/8 ours, 1.12-1.48x.**
- warp_spec zone (14 shapes, hpc's 3-WG persistent kernel): 0.69-0.99x; best at
  long-seq varlen ([512]×8 0.99, [512]×16 0.98, zipf 0.97).
- **Honest unified-run discovery**: v5 (stages=1) was the *best* version on the
  big-grid † band (rows 16-21: 0.95-1.03x) — v6's stages=2 default costs ~4-7%
  there (96 KB smem → 2 CTA/SM vs 3) while winning the small grids. A
  `pick_stages` rule (stages=1 for grids ≥ ~512 CTAs) is the identified
  follow-up, recorded in Closure.

### Key takeaways
- **Correctness frozen**: 22 shapes × 3-way (torch SDPA / hpc-ops / cutedsl) 0
  Failed at every version's re-bench; `run_prefill` 22/22 + `test_varlen` 5/5 at HEAD.
- **The journey is two phase transitions, not ten equal steps**: v1→v2 killed the
  load-latency chain (16x), v4→v5 killed the Q-serial critical path (2.6x) and
  flipped the stall regime; everything after v5 is shaving the *barrier-bound*
  single-WG regime (v6-v10: 1.28x→1.39-1.48x total).
- **Same knob, different verdict, depending on the critical path**: stages=2 was
  neutral at v3 (Q-load dominated), +42% at v6c (after Q-async), and −5% at v6 on
  big grids (occupancy > prefetch there). A/B every knob after changing the chain.
- **Negative results kept as infrastructure**: v4 split-KV ships OFF until v5 made
  it win on small grids (now automatic via pick_split ≤96 CTAs); v10's wait-removal
  was neutral but provably-safe simplification.
- ncu attribution chain (shape0): long_scoreboard 12131 → 710 → (plateau 3
  versions) → 174 → barrier-top 113/119/118/114 — each lever deleted exactly the
  stall ncu named; the terminal barrier+wait profile IS the single-warpgroup
  serialization = the declared warp-spec red line.

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

### Δ analysis (unified re-bench — baseline)
- **Effect**: 512² 1.008 ms, vs hpc 0.031x; the journey's worst row (1077 ms on
  [16384] — 69x behind hpc there).
- **Metric owned**: long_scoreboard **12131 samples** + wait 5670; tensor pipe
  0.03%. Every warp's life is waiting on global load returns.
- **Mechanism**: `autovec_copy` issues per-thread LDGs into register fragments;
  in-order register consumers (WGMMA operands, softmax) can't start until whole
  tiles land, and nothing overlaps load with compute, within or across tiles.

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

### Δ analysis vs v1 (unified)
- **Effect**: 1.008 → 0.070 ms = **16x**; vs hpc 0.031x → 0.44x. Biggest single
  jump until v5.
- **Metric moved**: long_scoreboard 12131 → 710 samples (−94%); Duration
  1490 → 86.3 µs; tensor 0.03 → 0.57%. What remains is `wait` (566): warps now
  mostly wait on each other, not on DRAM.
- **Mechanism**: TMA = one-thread descriptor-based bulk copy; the DMA engine
  moves 16 KB tiles asynchronously, mbarrier + tx-count exposes completion
  without register pressure. Load left the critical path — but the *loop* is
  still issue→wait→consume serial (stages=1 shipped), and **Q's autovec_copy
  stayed synchronous**: exactly the next two plateaus.

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

### Δ analysis vs v2 (unified)
- **Effect**: none — 0.070 → 0.071 ms, all three knobs ≤4%.
- **Metric moved**: nothing outside noise (long_sb 710↔842, tensor 0.57↔0.55%).
- **Mechanism of the neutrality**: occupancy, L2 order and mask-skip target
  *second-order* costs while the critical path is the serial per-iteration chain
  (sync Q load + wait-then-consume). Amdahl: you cannot tune what the bottleneck
  doesn't route through. This step's real product is the Step 3.5 prescription —
  'parallelism starvation + long_scoreboard 34%' — which named v5's lever two
  versions early.

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
| **(1,1,128,[512])** single head | 8 | 0.073 | 0.070 | 0.067 | **1.46x faster** |
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

### Δ analysis vs v3 (unified)
- **Effect**: fused default unchanged (0.071 ms, split-K OFF after the A/B); the
  experiment itself is the deliverable.
- **Metric moved (under split=2)**: main kernel 71→83 µs (+17%) with combine only
  4.2 µs — grid doubled, per-CTA work halved, wall time still *grew*.
- **Mechanism of the negative**: splitting the KV loop cuts the minority cost
  (iterations) but multiplies the majority cost (per-CTA fixed: sync Q load,
  first TMA wait, drain ×S) plus new partial traffic. Parallelism knobs cannot
  beat a fixed-latency-dominated chain — only shortening the chain can (v5).
  Infrastructure kept: the partial epilogue + combine became v6a's pick_split
  substrate.

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

### Δ analysis vs v4 (unified)
- **Effect**: 0.071 → 0.027 ms = **2.6x**; vs hpc crosses 1.0 for the first time
  (1.17x at 512²); all 22 shapes jump ≥2.3x.
- **Metric moved**: long_scoreboard 727 → 174; Duration 87.4 → 24.4 µs; tensor
  0.56 → 2.15%; **barrier appears at top-2 (112)** — the regime flip v6 attacks.
  cyc/inst 4.55 → 10.11 (fewer, longer stalls: the load herd is gone, pipeline
  waits remain).
- **Mechanism**: Q rides its own single-stage `PipelineTmaAsync` issued in the
  prologue — the 16 KB LDG chain that *every* iteration's first WGMMA waited on
  becomes one async copy overlapping barrier init + first KV prefetch, with
  consumer_wait placed right before the first MMA. Direct execution of the
  Step 3.5 prescription.

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

### Δ analysis vs v5 (unified)
- **Effect**: 512² fused 0.027 → 0.023 ms (1.28x); s2-eligible small grids reach
  1.4x+ (four-quadrant matrix above).
- **Metric moved**: barrier 113 becomes the **top** stall for the first time;
  tensor 2.15 → 2.67%; Duration 24.4 → 19.1 µs.
- **Mechanism**: (b) sP's 8 KB smem deleted (compile-time template only); (c)
  stages=2 ring *turned positive only because v5/v6b removed the latency it must
  hide* — the same knob was neutral at v3 (A/B after every chain change);
  (a) pick_split ≤96 CTAs encodes 'split iff the GPU is under-filled'.
- **Unified-run addendum (honest)**: the four-quadrant matrix only covered the 8
  small shapes; on big grids stages=2 costs 4-7% vs v5 (96 KB → 2 CTA/SM instead
  of 3; matrix rows 16-21 peak at v5). `pick_stages` recorded in Closure.

## Step 7: ✅ v7 — vectorized LSE combine + sO dead-smem removal (small grids to 1.03-1.46x)

**tag: `flash-ex1-v7-combine`** — three follow-ups from the v7 option list:

1. **combine_kernel vectorized**: was 1 thread = 1 column with scalar 4B loads
   (grid T×H, block 128). Now `VEC=4` columns/thread (16 B `autovec_copy` loads of
   PO + vectorized bf16 store) and `ROWS=4` rows/CTA (block `(Dd/4, 4, 1)`). On the
   split=2 shapes this moved e.g. GQA×3 0.028→0.026 ms, [512,768] 0.034→0.032 ms;
   512² itself sat *at* the combine's launch floor (the 4.2 µs combine overlaps
   other CTAs' mainloop tail, so its cost was partly hidden) — net: honest −4-7%
   where combine is on the critical path, elsewhere noise.
2. **sO MemRange deleted (−16 KB/CTA)**: like sP in v6b, `sO` turned out to be a
   pure `partition_C` layout template — `tCrO` lives in registers and the epilogue
   copy (`r2s_tiled_copy_o`, misleadingly named) goes reg→gmem directly. Both
   sP/sO now borrow sV's base pointer; smem 96→80 KB/CTA at stages=2. Occupancy
   stays at 2 CTA/SM (3 would need ≤75 KB) and long-seq numbers are unchanged
   (within noise), so this is pure memory headroom, not a speed lever.
3. **pick_split threshold kept at 96** (data correction from the review): at
   exactly 96 CTAs the two shapes disagree — unequal [512,768] *gains* +0.16x from
   split=2 while balanced GQA×3 *loses* −0.04x — so ≤96 stays; documented in the
   docstring.

### Result (v7 = current default)
multi-stage zone **8/8 vs hpc-ops, 1.03-1.46x** (H1 1.46, 512² 1.42, GQA512 1.40,
[512,768] 1.37, GQA×3 1.33, [200,328] 1.12, [512]×4 1.08, 1024² 1.03); Llama3
standard zone unchanged 0.79-0.96x (zipf varlen 0.96x).

### Verified
- `make quality` ✓; `run_prefill.py` **22/22**; `test_varlen.py` **5/5**;
  `compare_hpcops.py` small-8 + Llama3/varlen-8 **3-way 0 Failed** each.

---

### Δ analysis vs v6 (unified)
- **Effect**: fused shape0 19.1 → 19.2 µs (unchanged — this step doesn't touch
  the fused path); the wins are on split=2 shapes where combine is on the
  critical path (−4-7%, e.g. GQA×3 0.028→0.026 ms).
- **Metric moved**: inside combine — scalar 4B → 16B vector loads, 4 rows/CTA;
  the main kernel's ncu signature is identical by construction (tensor 2.67→2.63%).
- **Mechanism**: sO is a second never-dereferenced partition_C template (the
  epilogue is reg→gmem despite the r2s name) → −16 KB pure headroom. 512² sitting
  *at* the combine launch floor (4.2 µs partly hidden by CTA-tail overlap) is
  itself the evidence that combine isn't the wall there.

## Step 8: ✅ v8 — O epilogue TMA store (fused shapes to 0.89-0.96x in the Llama3 zone)

**tag: `flash-ex1-v8-tma-store`** — the plan's "small tail-latency win" turned out to
be a real win on medium-batch fused shapes and a wash elsewhere.

### Change
The fused (split_k==1) epilogue was a register→gmem scattered `CopyUniversalOp`
store. Now, lesson-11 style: cast→`r2s_tiled_copy_o` into smem→`fence_proxy
("async.shared", space="cta")`→`sync_threads`→warp0 `elect_one` issues one
`CopyBulkTensorTileS2GOp` bulk store→`cp_async_bulk_commit_group` +
`wait_group(0, read=False)` before CTA exit. The sO landing pad **aliases the sQ
region** (Q's last read is the final QK WGMMA, ordered by the sync) — so smem
stays 80 KB/CTA, no new allocation. `__call__` builds `o_view (S, D, H_q)` +
`tma_atom_o/tma_tensor_o` exactly like q_view; the kernel's `mO` param becomes
`(tma_atom_o, mO_tma)`. The split=2 partial path keeps its direct reg→gmem PO
store (fp32 tile wouldn't fit the sQ alias).

Two debugging lessons now baked into comments:
- a TMA box that covers the whole smem tile leaves **no rest-mode** on the smem
  side: the store coord is `tOsO[None]`, not `[None, 0]`;
- r2s-write and TMA-read disagreed on the **swizzle** (one XORed, one didn't →
  paired-element scrambling past d≥64); sO is now plain row-major on *both* sides.

### Result vs v7 (same-run hpc ratios; 3-way 0 Failed, 22 shapes)
- multi-stage zone: unchanged within noise (1.30-1.47x; 1024² 1.03→1.09,
  [512]×4 1.08→1.13 — the two auto-split=1 small shapes got the store win).
- †/Llama3 fused-heavy zone: **[512]×32 0.79→0.92x (0.826→0.709 ms, −14%)**,
  [1024]×16 0.85→0.91, [512]×8 →0.96, [512]×16 →0.94, zipf 0.96, U-dist 0.94;
  the first real gains against hpc's warp_spec kernel.
- Cost: longest sequences give back ~3-4% ([8192] 0.92→0.89, [16384] 0.92→0.89)
  — initially blamed on the extra sync + bulk-wait before CTA retire, but Step 10
  removed the wait with zero change: it was measurement drift.
  Net strongly positive on the fused-dominant middle band; accepted.

### Verified
- `make quality` ✓; `run_prefill.py` **22/22**; `test_varlen.py` **5/5**;
  compare small-8 + big-14 **3-way 0 Failed**; fused spot-check FA_SPLIT=1 shape0
  Success 0.00195.

---

### Δ analysis vs v7 (unified)
- **Effect**: 17.6 µs (−8% on shape0); the fused-heavy big-grid band recovers
  most of the v6 stages=2 tax ([512]×32 0.79→0.92, [512]×16→0.96, zipf→0.96).
- **Metric moved**: long_scoreboard 54 (scatter stores gone); LSU pipe down,
  TMA pipe carries the 16 KB O tile out; DRAM 2.25%.
- **Mechanism**: register→smem into the **sQ-aliased** pad (zero new smem: Q is
  fully consumed by the last QK WGMMA before the epilogue sync) → one bulk S2G
  store. DSL lessons preserved in code comments: a full-tile TMA box leaves no
  smem rest-mode (`tOsO[None]`), and r2s vs TMA must agree on swizzle (plain
  row-major both sides after d≥64 scrambling).

*(No Step 9 / v9 exists: BLK_N=128 was rejected by analysis before being
coded — QK would stay two serial N=64 WGMMA and softmax volume is unchanged,
so the critical path cannot shorten, while the cost would be a pad-128 contract
change plus 144 KB smem (1 CTA/SM). The number was left as a hole on purpose.)*

## Step 10: ➖ v10 — drop the epilogue bulk commit/wait (neutral, kept as simplification)

**hypothesis**: the v8 fused path's ~3-4% regression on [8192]/[16384] came from
`cp_async_bulk_commit_group()` + `wait_group(0, read=False)` delaying CTA retire.
**Result: falsified.** With an idle GPU (an unrelated co-tenant had been inflating
*both* kernels ~2x in the first measurements — caught by hpc's own 7.9→15.2 ms
baseline), removing the wait leaves everything at v8 levels: [8192] 8.815 ms
(0.89x), [16384] 17.519 ms (0.89x), [512]×8 0.041 (0.97x), and the 8 multi-stage
shapes 1.09-1.47x, all within noise of v8. The v8 "regression" was measurement
drift, not the wait.

Kept anyway: the wait was provably redundant — PTX guarantees outstanding bulk
async groups complete before kernel exit, and the fused CTA never rewrites the
sO pad after issuing the store — so this removes one barrier and two instructions
with zero downside.

### Verified
- `run_prefill.py` **22/22**, `test_varlen.py` **5/5**, clean-GPU compare long-5 +
  small-8 **3-way 0 Failed**.

---

### Δ analysis vs v8 (unified)
- **Effect**: neutral (17.6→17.8 µs; matrix v8→v10 within ±2% everywhere, 12/22
  shapes slightly up). The v8-era '[8192]/[16384] −3%' narrative retires — the
  unified re-bench attributes those rows to the v6 stages=2 occupancy tax (v5
  still leads them; Step 6 addendum).
- **Metric moved**: none by design.
- **Mechanism**: the removed `commit_group`/`wait_group(0, read=False)` was
  provably redundant — PTX guarantees outstanding bulk groups complete before
  kernel exit, and the fused CTA never rewrites the sO pad after issuing. Kept
  as a one-barrier-fewer simplification.

## Step 11: ✅ v11 — programmatic dependent launch on the split combine

**tag: `flash-ex1-v11-pdl`** — first post-freeze gain patch (v10 remains the
`flash-ex1-v10-frozen` anchor).

### Change (10 lines, split path only)
- combine `.launch(..., use_pdl=True)` → CUDA attribute
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`: the combine grid
  launches (blocks scheduled, prologue runs) *while the main kernel is still
  finishing*, instead of queueing serially on the stream.
- `cute.arch.griddepcontrol_wait()` at combine entry: blocks there until the
  previous grid is complete AND its writes flushed — the PO/Pm/Pl reads that
  follow are guaranteed visible. Pairing is load-bearing: `use_pdl` without the
  wait races; the wait without `use_pdl` is a no-op (both sides commented).
- The main kernel is untouched: its ordinary completion is the implicit
  release. (The producer-side `launch_dependents()` early-release is the
  documented optional extension — skipped: it releases while PO-writing CTAs
  are still resident, and the SM-resource trade was not worth testing blind.)

### Δ analysis vs v10 (unified harness, dual-run, split=2 band)
- **Effect**: −4-14% wall on every split=2 shape — 512² 0.022→0.020 (−9~14%),
  GQA512 →0.019 (−14%), H1 →0.018 (−10%, new best 1.61x vs hpc), [512,768]
  →0.031 (−6~9%), GQA×3 →0.025 (−4%); [200,328] +5% is the sole regression
  (its main kernel is so short there's little tail left to overlap).
  **split=1 controls 1024²/[512]×4 moved 0~+3% = noise, confirming the gain is
  PDL-attributed, not run drift.** vs hpc on the band: 1.19-1.61x.
- **Metric moved**: inter-kernel gap in the stream — the combine's ~4 µs launch
  latency sinks under the main kernel's CTA-drain tail (this was exactly the
  'combine launch floor' identified in Step 7's Δ analysis).
- **Mechanism**: PDL overlaps grid launch/setup with predecessor execution;
  `griddepcontrol.wait` keeps the memory-order contract. DSL support:
  `LaunchConfig.use_pdl` (base_dsl/dsl.py) + `cute.arch.griddepcontrol_wait`
  (arch/nvvm_wrappers.py) — Closure lever 3's 'not wrapped' note was wrong and
  is struck.

### Verified
- `run_prefill.py` **22/22**, `test_varlen.py` **5/5**, compare 8 shapes × 2
  runs 3-way **0 Failed**.

---

## Step 12: ❌ v12 — single-kernel split-KV via Hopper cluster + DSMEM (built, measured, reverted)

**no feature tag (rejected experiment; v11 remains terminal)** — user-chosen
route B: replace the two-kernel split path (main + LSE-combine) with a
**2-CTA cluster** (`cluster=(1,1,2)`): each CTA computes one KV span, writes its
un-normalized fp32 partial + (m, l) into its own `cutlass.Array` smem buffers, a
non-relaxed `barrier_cluster_arrive/wait` pair publishes them, then both CTAs
mapa-**pull** the peer's partial, do the LSE merge in-register, and rank 0 emits
the v8 TMA store. combine kernel, PO/Pm/Pl gmem buffers and PDL all deleted
**in the experiment only** — mainline reverted with the experiment and still
ships the v11 two-kernel split with PDL (Step 11).

### Engineering outcome: every primitive works; the idea doesn't
- Correctness fully achieved: fused regression 22/22, cluster path 22/22,
  test_varlen 5/5, compare 3-way 0 Failed — the DSL *can* build a
  DSMEM-merged single-kernel split attention. Key findings (see below).
- Performance: split-band **regressed +0~35%** vs v11 — 512²/GQA512/H1 roughly
  flat-to-+11%, multi-batch shapes brutally worse ([512,768] 0.031→0.042,
  [200,328] 0.021→0.027 (falls below hpc!), GQA×3 0.025→0.033). Reverted.

### Why it loses (measured attribution)
1. **Merge parallelism collapse dominates**: v11's combine runs as its own grid
   (~2048 CTAs across the whole GPU) and v11's PDL already hid its launch gap.
   v12 squeezes the same LSE math into the 128 threads of one cluster member —
   scalar per-element mapa loads (64/thread) are far slower than the massively
   parallel dedicated kernel. Saving ~4 µs of launch pays nothing when the
   merge itself costs +6-11 µs.
2. **Cluster gang-scheduling + smem tax on exactly the multi-batch shapes**:
   the split path's smem grew 80 → ~112.5 KB/CTA (sPart 32 KB + stats), hugging
   the 2-CTA/SM ceiling, and both members must co-reside in one GPC — fine for
   tiny grids, harmful at 96-192 CTA (the +30% band).
3. Lesson generalized: **co-location ≠ free**. Turning a launch boundary into a
   cluster boundary imports the scheduler's constraints into the hot path; it
   only pays when the merged stage is small AND latency-bound (here it was small
   but wide-parallel after PDL).

### Reusable findings banked (from X1-X4 bisection + fix)
- Full cluster recipe verified in a real kernel: `launch(cluster=(1,1,2))`,
  `block_idx_in_cluster()`, `cutlass.Array(fp32,(64,128),smem,16)` coexisting
  with `@cute.struct` SharedStorage + `PipelineTmaAsync(cta_layout_vmnk=
  (1,1,1,1))` (X1: innocent), non-relaxed arrive/wait for publication,
  `prims.mapa(ptr, peer_rank)` + `(peer+off).load()` for pulls, and a terminal
  `arrive_relaxed+wait` so no member frees smem while its peer reads (X-era
  crash #1).
- **Identity-tensor coordinate trap** (crash #2, illegal access): `c_idx`-derived
  row numbers are *batch-local absolute* — correct for v4's gmem Pm/Pl writes,
  OOB by up to 448 when indexing a 64-slot tile-local smem buffer. Fix: subtract
  `q_tile_start`. Rule: re-audit the coordinate space whenever a c_idx-derived
  index changes destination.
- DSL `map_dsmem_ptr`'s dsmem tensors don't support load/store —
  `cutlass.experimental.primitives.mapa` is the only working pull path.

### Verified (as experiment, on /tmp/v12_backup.py)
22/22 ×2 configs, 5/5, 3-way 0 Failed; bench table in journey row above. Code
reverted to v11 (04535ea); no tag.

---

## Closure: frozen state & unused levers (v11 = final; frozen anchor v10)

**This kernel line was frozen at v10; v11 PDL is the single post-freeze gain
now shipped.** Final per-version evidence lives in the
Master Performance Table (unified 9-version × 22-shape matrix + ncu chain), and
each Step's `Δ analysis` block ties its lever to the metric it moved.

### Identified-but-unimplemented (ranked by measured evidence)
1. **`pick_stages`** — the unified matrix's clearest open win: stages=1 beats
   stages=2 by 4-7% on big grids (rows 16-21 peak at v5) while stages=2 wins
   small grids; a grid-size rule would take both. Cost: one constant + re-verify.
2. **Warp specialization** (load WG + compute WGs, hpc's warp_spec / CUTLASS
   FMHA): the v10 stall profile (barrier+wait top, 1.00 active warp/sched) names
   it as the *only* remaining big lever — declared out of scope (red line; it is
   a different kernel, ex.2).
3. ~~PDL between main + combine~~ **IMPLEMENTED as v11** (−4-14% on the split
   band; `use_pdl` + `griddepcontrol_wait` are both wrapped by the DSL — the
   original 'not wrapped' note was wrong). Producer-side `launch_dependents()`
   early-release remains unexplored.
4. **TMA multicast / clusters+DSMEM**: clusters+DSMEM as a merge vehicle was
   built and REJECTED (Step 12: +0~35%, merge-parallelism collapse). TMA
   *multicast* itself is now PROVEN in this repo — the gemm line shipped it
   with SASS-level verification (`ops/gemm/PERFLOG.md` #18, tag
   `gemm-v9-multicast`) — but it measured neutral there too (compute-bound
   GPU, L2-resident operands). Applying it to FA KV tiles (rows 16-21) remains
   untried and inherits the same no-gain expectation on H20 unless the working
   set exceeds L2.
5. **L2 persistence window** (`cudaAccessPolicyWindow`) for K/V prefixes on
   serving rows ([512]×8..×32).
6. **FP8 P·V / reduced softmax precision**: hpc's own D-series direction; breaks
   this exercise's bf16+fp32 correctness envelope, out of scope.
7. **PV-side deeper pipelining** (double-buffered accumulators, QK(it+1) ∥
   PV(it)): needs two WGs under Hopper WGMMA semantics — folds into lever 2.
8. **exp2/MUFU tuning**: XU 1.57% active at v10 — measured irrelevant.
9. **Hand-scheduled PTX / C++ CuTe**: DSL codegen quality is a known-unknown
   (hpc's >100%-peak figures are the same flops convention, so ratios are the
   honest metric); outside this exercise's toolkit.
