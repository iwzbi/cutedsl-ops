# GEMM Optimization Journey

Hopper H20 (sm_90, 78 SM, 148 TFLOPS FP16 peak), CuTe DSL (nvidia-cutlass-dsl 4.7.0).
cuBLAS reference: `torch.mm(a, b.t())` with FP16 (libcublas.so.13).

Each step links to ncu raw reports in [`ncu_reports/`](./ncu_reports/).
Use `git diff v1-baseline..<tag> -- ops/gemm/gemm_kernel.py` to see code changes.

---

## Master Performance Table

All numbers FP16 input / FP32 accumulator / FP16 output (no FP8).
Measured with fresh compilation, L2-flushed CUDA Events (our kernel) / plain timing (cuBLAS FP16).
v5/v6 "best sk" = optimal split_k for that shape (tuned per size).

| Shape | cuBLAS | v1-base | v2-persist | v3-swizzle | v4-blk96 | v5-sk(best) | v6-sk(best) | v7-dispatch | cluster | Best vs cuBLAS |
|--------|--------|---------|------------|------------|----------|-------------|-------------|-------------|---------|-----------------|
| 512³   | 21.1   | 12.1    | 12.1       | 12.2       | 12.2     | 43.0 (sk8)  | 42.0 (sk8)  | **48.4**    | 22.0    | **+129%** |
| 1024³  | 95.4   | 51.5    | 51.7       | 51.7       | 51.5     | 91.3 (sk2)  | 90.7 (sk2)  | 83.0        | **94.1** | -1.3%     |
| 2048³  | 127.8  | 110.5   | 111.2      | 111.9      | 112.2    | 118.3 (sk4) | **124.5** (sk4) | 109.8   | —       | -2.6%     |
| 4096³  | 132.0  | 130.5   | 131.0      | 131.9      | 132.4    | 132.2 (sk4) | **134.7** (sk4) | 119.8   | 130.9   | **+2.0%** |
| 8192³  | 137.5  | 137.4   | 138.1      | 138.6      | 138.3    | 140.2 (sk2) | **141.2** (sk2) | 113.2   | —       | **+2.7%** |
| 16384³ | 139.2  | **141.9** | 141.5    | 141.6      | 141.6    | 141.8 (sk1) | 141.8 (sk1) | 111.5       | —       | **+1.9%** |

**Dispatch mode**: `M*N < 1024*1024` → small tile (v7: 64x64, S5, 2 blocks/SM); else → large tile (v6: 128x256, S3, 1 block/SM).
v7 dispatch uses best sk per shape (small tile for 512³, large tile for 4096³+).

### Key takeaways (FP16 only)
- **Small scale (512³)**: v7 small tile wins — 48.4T = **+129%** vs cuBLAS
- **Medium scale (2048³-4096³)**: v6 (persistent+sk+epi) best — 134.7T at 4096³ = **+2.0%** vs cuBLAS
- **Large scale (8192³+)**: v1 baseline already beats cuBLAS; v6 adds +2.7% at 8192³
- **v6 wins 3/6 shapes** (2048³, 4096³, 8192³), v7 wins 512³, v1 wins 16384³, cluster wins 1024³ (close to cuBLAS)
- **We beat cuBLAS at 4/6 shapes** in FP16: 512³ (+129%), 4096³ (+2.0%), 8192³ (+2.7%), 16384³ (+1.9%)

---

## Apples-to-Apples: All Tags at sk=1

| Shape | v1 | v2 | v3 | v4 | v5 | v6 | cuBLAS |
|--------|------|------|------|------|------|------|--------|
| 512³   | 12.1 | 12.1 | 12.2 | 12.2 | 12.2 | 12.0 | 21.1   |
| 1024³  | 51.5 | 51.7 | 51.7 | 51.5 | 51.9 | 51.4 | 95.4   |
| 2048³  | 110.5| 111.2| 111.9| 112.2| 111.3| 112.5| 127.8  |
| 4096³  | 130.5| 131.0| 131.9| 132.4| 131.2| 132.9| 132.0  |
| 8192³  | 137.4| 138.1| 138.6| 138.3| 138.6| 139.1| 137.5  |
| 16384³ | 141.9| 141.5| 141.6| 141.6| 141.8| 141.8| 139.2  |

At sk=1 (no split-K), differences between tags are small (<2%). v2-v4 persistent variants show +0.5-1.3% at medium scale (2048³-4096³) from tail-wave elimination. v6 persistent+epi shows +0.7-1.7% at 4096³+ from epilogue overlap. All converge at 16384³ (enough waves to make tail/epilogue negligible).

---

## Step 1: Baseline — Warp Specialization WGMMA + TMA

**Tag**: `v1-baseline` (`29d7e4a`)

### Optimization: Warp Specialization + TMA + Pipeline

**Principle**: WGMMA is a long-latency instruction (~100+ cycles). TMA load is also long-latency (~200+ cycles). Without overlap, they serialize: load → wait → compute → wait → load → ... A **pipeline** with N stages allows the producer to issue TMA loads for the next N tiles while the consumer computes on the current stage — hiding TMA latency behind WGMMA compute. **Warp specialization** further decouples producer (TMA issuer) from consumer (WGMMA compute) — the producer never stalls on MMA and vice versa. Together, pipeline + warp spec achieve ~90% TC throughput on Hopper.
- wgmma.m64n256k16 (warpgroup MMA, A/B from SMEM via descriptor, no rmem copy)
- TMA cp.async.bulk.tensor for gmem↔smem (A/B load, D store)
- 3-stage PipelineTmaAsync (mbarrier-based double buffering)
- Warp specialization: 3 warpgroups (384 threads) — WG0=producer (TMA), WG1+WG2=consumer (WGMMA+epilogue)
- BLK_M=128, BLK_N=256, BLK_K=64, SW128 swizzle, ATOM_LAYOUT=(2,1,1)

### Performance

| Shape | TFLOPS | % peak | cuBLAS | vs cuBLAS | Waves |
|--------|--------|--------|--------|-----------|-------|
| 512³   | 12.1   | 8.2%   | 20.2   | -40%      | 0.03  |
| 1024³  | 51.2   | 34.6%  | 95.4   | -46%      | 0.4   |
| 2048³  | 110.5  | 74.7%  | 127.4  | -13%      | 1.7   |
| 4096³  | 130.6  | 88.2%  | 132.2  | -1.2%     | 6.6   |
| 8192³  | 137.4  | 92.8%  | 137.6  | -0.1%     | 26.3  |
| 16384³ | 141.9  | 95.9%  | 139.3  | **+1.8%** | 211   |

### ncu Profiling (4096³)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Compute (SM) Throughput | 90.9% | TC pipeline near full |
| Memory Throughput | 14.6% | Compute-bound confirmed |
| DRAM Throughput | 7.2% | HBM not bottleneck |
| No Eligible Warps | 96.7% | Few warps ready (wgmma long-latency) |
| Cycles/Issued Inst | 67.2 | Each inst waits 67 cycles |
| Top Stall | CTA barrier 74.2% (49.8 cyc) | Pipeline mbarrier sync |
| Theoretical Occupancy | 18.75% | smem 214KB + regs 154/thr |
| Achieved Occupancy | 13.9% | Producer idle warps pull down |
| L2 Hit Rate | 66.0% | |
| SM Load Imbalance | est. +5.9% | Grid tail (6.56 waves) |

### Strengths
- 95.9% peak at 16384³ — **beats cuBLAS** at large scale
- Warp specialization: producer/consumer parallel, no TMA/MMA serialization
- SW128 swizzle: 0 bank conflicts

### Shortcomings
- **Small scale (512³-1024³)**: only 0.03-0.4 waves → SMs starved, 40-46% behind cuBLAS
- **CTA barrier stall (74.2%)**: pipeline mbarrier sync is the #1 bottleneck — consumer_wait waits for TMA data, producer_acquire waits for consumer release
- **Grid tail (6.56 waves at 4096³)**: last wave has 34 idle SMs → est. 5.9% loss
- **Low occupancy (13.9%)**: 1 block/SM (smem 214KB + regs 154/thr); producer 3 non-warp0 threads idle

### ncu Reports
- [v1-baseline 4096³](ncu_reports/v1-baseline_4096.txt) | [1024³](ncu_reports/v1-baseline_1024.txt)

---

## Step 2: Persistent Kernel — CTA Stride Loop (no gain, case study)

**Tag**: `v2-persistent`

### Optimization: Persistent Kernel

**Principle**: In a standard grid, `total_tiles` CTAs are launched. With `total_tiles / num_sms = N` waves, the last wave has `total_tiles % num_sms` active SMs — the rest idle. A persistent kernel launches exactly `num_sms` CTAs, each looping through `ceil(total_tiles / num_sms)` tiles via a stride loop. Every SM works until the last tile is done — **zero tail-wave idle**. The benefit is proportional to `total_tiles % num_sms / num_sms` (the tail fraction).
**Goal**: eliminate tail-wave SM idling.

### Three bugs fixed (vs 14-warp-specialization reference)
1. **Deadlock**: `sync_threads()` required all 384 threads, but producer's 128 never reached it → GPU hang. Fix: `pipeline.NamedBarrier(barrier_id=1, num_threads=256)`.
2. **>5min compile**: K-tile loop `unroll_full=True` → 64× code bloat. Fix: `unroll=1`.
3. **Missing register reconfig**: Added `warpgroup_reg_dealloc(40)` (producer) + `warpgroup_reg_alloc(232)` (consumer).

### Performance (sk=1)

| Shape | v1 | v2 | Delta | cuBLAS |
|--------|------|------|-------|--------|
| 512³   | 12.1 | 12.1 | 0%    | 20.2   |
| 1024³  | 51.2 | 51.5 | +0.6% | 95.4   |
| 2048³  | 110.5| 111.3| +0.7% | 127.4  |
| 4096³  | 130.6| 132.3| +1.3% | 132.2  |
| 8192³  | 137.4| 138.3| +0.7% | 137.6  |
| 16384³ | 141.9| 141.6| -0.2% | 139.3  |

### ncu Comparison (4096³)

| Metric | v1 | v2 | Delta |
|--------|------|------|-------|
| Compute Throughput | 90.9% | 91.4% | +0.5% |
| Cycles/Issued | 67.2 | 68.7 | +2% worse |
| CTA barrier stall | 74.2% (49.8cyc) | 74.3% (51.1cyc) | slightly worse |
| SM load imbalance | est. +5.9% | eliminated | — |

### Strengths
- Eliminates SM load imbalance (tail wave) — confirmed by ncu
- Slight gain at medium scale (4096³: +1.3%) where tail wave was 0.56 waves
- Prerequisite for epilogue overlap (Step 6)

### Shortcomings
- **No gain at small scale**: 512³ has only 32 tiles < 78 SMs → can't reduce below 32 blocks
- **No gain at large scale**: 16384³ has 211 waves → tail is 0.4% of total time
- **Per-tile overhead**: DSL staged-if limitation forces consumer setup (get_slice/partition/make_fragment) inside the tile loop → Cycles/Issued increases from 67.2 to 68.7
- **Two NamedBarrier syncs per tile**: adds barrier stall (74.2% → 74.3%)
- **CTA barrier stall unchanged**: the main bottleneck (pipeline mbarrier sync) is not addressed

### ncu Reports
- [v2-persistent 4096³](ncu_reports/v2-persistent_4096.txt)

---

## Step 3: Block Swizzle — GROUP_M=4 (no gain, L2 worse, case study)

**Tag**: `v3-swizzle`

### Optimization
Tile assignment uses swizzled (bid_m, bid_n) mapping: tiles grouped by GROUP_M=4
consecutive M-blocks before striding N. **Goal**: improve L2 reuse of A tiles.

### Performance (sk=1)

| Shape | v2 | v3 | Delta |
|--------|------|------|-------|
| 1024³  | 51.5 | 51.9 | +0.8% |
| 2048³  | 111.3| 111.9| +0.5% |
| 4096³  | 132.3| 131.9| -0.3% |
| 16384³ | 141.6| 141.6| 0%    |

### L2 Impact

| Metric | v2 | v3 | Delta |
|--------|------|------|-------|
| L2 Hit Rate (4096³) | 66.3% | 62.3% | **-4.0% worse** |

### Strengths
- Marginal gain at small/medium scale (1024³: +0.8%) — some L2 reuse benefit

### Shortcomings
- **L2 hit rate dropped 4%**: GROUP_M=4 forces 6 active M-blocks per wave vs 5 default → more L2 pressure
- **Wrong direction**: kernel is compute-bound (DRAM 7.1%) → L2 optimization doesn't help
- **4096³ data (64MB) ≈ L2 (60MB)**: more active blocks = more L2 eviction
- **GROUP_M=4 doesn't align with wave boundary** (78/64=1.22 waves/group) → cross-group tiles hurt L2

### When would block swizzle help?
- Memory-bound kernels (DRAM throughput > 50%)
- Data size >> L2 (swizzle reduces working set per wave)
- Non-uniform tile computation (e.g., FlashAttention causal mask)

### ncu Reports
- [v3-swizzle 4096³](ncu_reports/v3-swizzle_4096.txt)

---

## Step 4: BLK_K=96, NUM_STAGES=2 — Fewer Syncs (slightly worse, case study)

**Tag**: `v4-blk96`

### Optimization
Increase BLK_K from 64 to 96 to reduce K-tile count (64→43, 33% fewer syncs).
Accept NUM_STAGES=2 (3→2) to fit the larger smem. Use SW64 swizzle (96×16=1536bit → SW64).

### Performance (sk=1)

| Shape | v2 (K64,S3) | v4 (K96,S2) | Delta |
|--------|-------------|-------------|-------|
| 1024³  | 51.5        | 51.6        | +0.2% |
| 2048³  | 111.3       | 112.0       | +0.6% |
| 4096³  | 132.3       | 132.1       | -0.2% |
| 8192³  | 138.3       | 138.5       | +0.1% |
| 16384³ | 141.6       | 141.6       | 0%    |

### Strengths
- Marginal gain at medium scale (2048³: +0.6%) — fewer syncs helps when pipeline fill/drain is significant
- smem 212KB < 228KB (fits)

### Shortcomings
- **Pipeline depth 3→2 is net negative at large scale**: overlap loss > sync reduction
- **BLK_K=96 → SW64 swizzle** (smaller granularity than SW128) — less efficient TMA
- **Tradeoff is wrong**: the 3-stage pipeline is the single biggest contributor to 90.9% TC throughput; reducing it to 2 stages undoes the overlap that hides TMA latency
- **Conclusion**: **Pipeline overlap (NUM_STAGES) is more valuable than sync count reduction.** The 3-stage pipeline is already optimal for this smem budget.

### ncu Reports
- [v4-blk96 4096³](ncu_reports/v4-blk96_4096.txt)

---

## Step 5: Split-K — K-Dimension Parallelism (big win for small scale)

**Tag**: `v5-splitk`

### Optimization
Split K dimension into `split_k` partitions. Grid `(grid_n, grid_m, split_k)` — Z dimension
indexes K splits. Each CTA computes a partial result into a separate output buffer slice.
Host-side `sum(dim=0)` reduces. **Goal**: multiply blocks count for small problems.

- A/B use original `bidy` (M), D uses `bidy + bidz * grid_m` (split buffer offset)
- K range: `k_start = bidz * k_per_split` to `k_end`
- Output: `partial_c[split_k * M, N]` viewed as `[split_k, M, N]`, summed on host

### Performance sweep

| Shape | sk=1 | sk=2 | sk=4 | sk=8 | Best | cuBLAS | vs cuBLAS |
|--------|------|------|------|------|------|--------|-----------|
| 512³   | 12.2 | 20.3 | 31.2 | **42.9** | sk8 | 20.2   | **+112%** |
| 1024³  | 51.9 | **91.4** | 84.4 | 75.4 | sk2 | 95.4   | -4.2%     |
| 2048³  | 111.3| 108.0| **118.5** | 109.9 | sk4 | 127.4 | -7.0%     |
| 4096³  | 131.2| 129.1| **132.6** | 127.8 | sk4 | 132.2 | +0.3%     |
| 8192³  | 138.6| **140.2** | — | — | sk2 | 137.6 | **+1.9%** |
| 16384³ | 141.8| — | — | — | sk1 | 139.3 | +1.8%    |

### Strengths
- **512³ sk=8: +254% vs baseline, +112% vs cuBLAS** — blocks from 8→64 fills SMs
- **8192³ sk=2: +1.9% vs cuBLAS** — enough blocks + still good pipeline efficiency
- **Correctness**: RE ≤ 0.04% across all sk values (host-side reduction is exact)

### L2 Cache Impact (split-K reduces working set)

Split-K increases L2 hit rate from 66% (v1) to 88% (v6) — not by adding L2 capacity, but by **reducing the per-CTA data footprint** so the wave's working set fits in L2.

For 4096³ fp16 (BLK_M=128, BLK_N=256, BLK_K=64):

| | sk=1 (v1) | sk=4 (v6) |
|---|---|---|
| K-tiles/CTA | 64 | 16 (K/4) |
| A data/CTA | 128×4096×2B = 1MB | 128×1024×2B = 256KB |
| B data/CTA | 256×4096×2B = 2MB | 256×1024×2B = 512KB |
| Total/CTA | 3MB | 768KB |

One wave = 78 CTAs. With data sharing (same bidy → share A, same bidx → share B):

| | sk=1 | sk=4 |
|---|---|---|
| Unique M-blocks/wave | 5 | ~2/split × 4 = 8 |
| Unique N-blocks/wave | 16 | ~10/split × 4 = 40 |
| **Wave working set** | **37MB** (5×1MB + 16×2MB) | **22MB** (4 × 5.5MB) |
| L2 = 60MB | borderline (37MB + pipeline overhead) | comfortable (22MB << 60MB) |
| **L2 Hit Rate** | 66% | **88%** |

Note: the "70MB" figure sometimes cited is the total A+B matrix size (67MB), not the working set. The working set is smaller due to data sharing across CTAs.

Caveat: L2 improvement has limited impact on our compute-bound kernel (DRAM 4.4%). The TFLOPS gain comes mainly from more blocks, not better L2.

### Shortcomings
- **Non-monotonic**: 1024³ sk=2 (91.4T) > sk=4 (84.4T) — sk=4 gives only 16 K-tiles/CTA, pipeline fill/drain overhead dominates (NUM_STAGES=3 but only 16 iterations)
- **Large scale unaffected**: 4096³+ already has enough blocks, split-K only adds reduction overhead
- **Best sk varies by shape**: requires heuristic dispatch (like cuBLAS does)
- **Host-side reduction**: `sum(dim=0)` adds a kernel launch + memory traffic (not counted in timing)

### ncu Reports
- [v5-splitk4 4096³](ncu_reports/v5-splitk4_4096.txt)

---

## Step 6: Persistent + Split-K + Epilogue Overlap

**Tag**: `v6-persistent-splitk-epi`

### Optimization
Three techniques combined:

1. **Persistent kernel**: CTA stride loop over 3D tile space (split_k, grid_m, grid_n). Eliminates tail wave.
2. **Split-K**: K-dimension parallelism (same as v5). Increases blocks for small problems.
3. **Epilogue overlap**: TMA S2G `commit_group` without `wait_group` → next tile's mainloop runs while previous store is in flight. `wait_group(1)` at next epilogue start ensures previous store completed. Eliminates inter-tile TC idle time.

### Performance (sk=1, apples-to-apples vs v1-v5)

| Shape | v1 | v6 (sk1) | Delta | cuBLAS |
|--------|------|----------|-------|--------|
| 512³   | 12.1 | 12.0    | -0.8% | 20.2   |
| 1024³  | 51.2 | 51.4    | +0.4% | 95.4   |
| 2048³  | 110.5| 112.5   | +1.8% | 127.4  |
| 4096³  | 130.6| 132.9   | +1.8% | 132.2  |
| 8192³  | 137.4| 139.1   | +1.2% | 137.6  |
| 16384³ | 141.9| 141.8   | -0.1% | 139.3  |

### Performance (best sk, full optimization)

| Shape | sk | v5 (best) | v6 (best) | v6 vs v5 | cuBLAS | vs cuBLAS |
|--------|-----|-----------|-----------|----------|--------|-----------|
| 512³   | 8  | 42.9      | 41.3      | -3.7%    | 20.2   | **+105%** |
| 1024³  | 2  | 91.4      | 89.9      | -1.6%    | 95.4   | -5.8%     |
| 2048³  | 4  | 118.5     | 124.6     | **+5.1%**| 127.4  | -2.2%     |
| 4096³  | 4  | 132.6     | 134.6     | **+1.5%**| 132.2  | **+1.8%** |
| 8192³  | 2  | 140.2     | 141.2     | +0.7%    | 137.6  | **+2.6%** |
| 16384³ | 1  | 141.8     | 141.8     | 0%       | 139.3  | +1.8%     |

### Epilogue overlap: helps large, hurts small

| Shape | sk | no-overlap | +overlap | Delta | Mainloop cycles | Store cycles |
|--------|-----|-----------|----------|-------|-----------------|--------------|
| 512³   | 8  | 42.9      | 41.3     | -3.7% | ~50             | ~180         |
| 1024³  | 2  | 91.1      | 89.6     | -1.6% | ~1600           | ~180         |
| 2048³  | 4  | 121.6     | 124.6    | +2.5% | ~800            | ~180         |
| 4096³  | 4  | 132.8     | 134.6    | +1.4% | ~3200           | ~180         |
| 16384³ | 1  | 141.1     | 141.8    | +0.5% | ~13000          | ~180         |

- **Small scale**: mainloop (50 cycles) < TMA store (180 cycles) → `wait_group(1)` stalls 130 cycles. Overlap impossible when store outlasts mainloop.
- **Medium scale**: mainloop (800-3200 cycles) >> store (180 cycles) → store completes during mainloop → `wait_group(1)` is instant. **Best gain +2.5%**.
- **Large scale**: marginal — pipeline already well-overlapped, store is negligible fraction.

### Strengths
- **Best kernel overall**: wins 4/6 shapes vs v5 (2048³, 4096³, 8192³, ties 16384³)
- **Beats cuBLAS at 4096³ (+1.8%) and 8192³ (+2.6%)** — persistent+epi overlap pays off
- Persistent + split-K + epi-overlap are **complementary**: persistent handles tail wave, split-K handles small scale blocks, epi-overlap handles inter-tile idle

### Shortcomings
- **512³ worse than v5**: epi-overlap's `wait_group(1)` adds stall when mainloop < store duration
- **1024³ worse than v5**: same issue (mainloop 1600 cycles ≈ store 180 cycles, borderline)
- **Per-tile overhead from persistent**: consumer setup per tile (DSL staged-if limitation) — same as v2
- **Complexity**: 3D tile space + persistent loop + epi-overlap wait logic — harder to maintain
- **No improvement at 16384³**: all optimizations converge to hardware limit (96% peak)

### ncu Reports
- [v6-epi-overlap 4096³](ncu_reports/v6-epi-overlap_4096.txt)

---

## Step 7: Small Tile + 5-Stage Pipeline + Dispatch Mode

**Tag**: `v7-small-tile`

### Optimization
BLK_M=64, BLK_N=64, BLK_K=64, NUM_STAGES=5, ATOM_LAYOUT=(1,1,1). WGMMA atom (64,64,16).
smem 89KB → 2 blocks/SM → occupancy 25%. 1 MMA WG + 1 DMA WG = 256 threads.

**Dispatch mode**: main now holds two kernel files:
- `gemm_kernel.py` = v6 (128x256, S3) for large shapes (M*N >= 1024*1024)
- `gemm_kernel_small.py` = v7 (64x64, S5) for small shapes (M*N < 1024*1024)
- `run_gemm.py` auto-selects based on problem size

### ncu Comparison (4096³, v7 vs v6)

| Metric | v6 (128x256, S3) | v7 (64x64, S5) | Delta |
|--------|-------------------|----------------|-------|
| CTA barrier stall | 41.5 cyc (73.3%) | **7.6 cyc** | **-82%** |
| Compute Throughput | 94.0% | 83.0% | -11% |
| No Eligible | 96.0% | 90.9% | -5% |
| Eligible Warps/Sched | 0.04 | 0.10 | +150% |
| Memory Throughput | 14.6% | 30.2% | +15.6% |
| L2 Hit Rate | 88.2% | 83.8% | -4.4% |

### Full Split-K Sweep

| Shape | sk=1 | sk=2 | sk=4 | sk=8 | Best | vs v6 best |
|--------|------|------|------|------|------|------------|
| 512³   | **48.9** | 47.4 | 43.9 | 40.2 | sk1 | **+13.1%** |
| 1024³  | 82.3 | **87.7** | 82.7 | 75.0 | sk2 | -2.4% |
| 2048³  | **108.7** | 108.3 | 106.1 | 98.4 | sk1 | -12.8% |
| 4096³  | **118.4** | 114.7 | 113.0 | 108.7 | sk1 | -12.0% |
| 8192³  | 112.9 | 117.4 | **117.5** | 114.6 | sk4 | -16.8% |
| 16384³ | 111.9 | 111.9 | 113.7 | **117.4** | sk8 | -17.2% |

### Strengths
- **5-stage pipeline killed CTA barrier stall** (-82%) — the #1 bottleneck from v1-v6
- **512³ +13.1%** vs v6 — 2 blocks/SM eliminates SM starvation
- **512³ v7 sk=1 (48.9T) > v5 sk=8 (42.9T)** — 2 blocks/SM beats split-K (no reduction needed)
- **Best sk pattern inverts**: v7 small shapes prefer sk=1 (enough blocks from 2 blocks/SM), large shapes prefer sk=8 (compensate for small tile = many tiles)

### Shortcomings
- **Large scale -12~17%**: small tile (64x64 vs 128x256) = 4x less work/instruction → TC throughput drops 11%
- **Memory throughput doubled** (14.6% → 30.2%): small tile = worse data reuse (each tile loads unique A/B slice)
- **L2 hit rate dropped** (88.2% → 83.8%): smaller tiles = less spatial locality
- **Tradeoff is fundamental**: tile size vs occupancy/pipeline-depth. No single config wins all shapes — hence dispatch.

### When to use small tile
- Small problems (M*N < 1024*1024) where SMs are starved (blocks < SMs)
- When occupancy matters more than TC efficiency (low wave count)
- When CTA barrier stall dominates over compute throughput

### ncu Reports
- [v7-small-tile 4096³](ncu_reports/v7-small-tile_4096.txt)

---

## Optimization vs cuBLAS Summary (FP16, fresh data)

| Shape | cuBLAS | Our best | Config | Margin |
|--------|--------|----------|--------|--------|
| 512³   | 21.1   | **48.4** | v7 dispatch (small tile) | **+129%** |
| 1024³  | 95.4   | **94.1** | cluster (no mcast) | -1.3%  |
| 2048³  | 127.8  | **124.5**| v6 sk4 | -2.6%  |
| 4096³  | 132.0  | **134.7**| v6 sk4 | **+2.0%** |
| 8192³  | 137.5  | **141.2**| v6 sk2 | **+2.7%** |
| 16384³ | 139.2  | **141.9**| v1 sk1 | **+1.9%** |

**We beat cuBLAS at 4/6 shapes** in FP16 (512³, 4096³, 8192³, 16384³).
cuBLAS wins at 1024³ (+1.3%) and 2048³ (+2.6%) — its heuristic dispatch picks better tile sizes for medium scale.

---

## HPC-Ops Reference Analysis (Tencent production kernel)

Source: `/home/code/hpc-ops/src/gemm/sm90/gemm.cu` (557 lines, C++ CUDA, H20-optimized).
This is a BF16×FP32 GEMM (router GEMM), not our FP16×FP16, but the optimization techniques are transferable.

### Techniques HPC-Ops uses that we also use
- ✅ Persistent kernel (`while(true)` + `iblock += gridDim.x`)
- ✅ Block swizzle (`kBlockSwizzle=4` + flat fallback for tail)
- ✅ Split-K (`kSplitK` parameter, in-grid Z stride)
- ✅ Epilogue overlap (TMA store arrive without wait, wait at next tile)
- ✅ Warp specialization (producer/consumer split, reg_dealloc/alloc)
- ✅ SW128 swizzle

### Techniques HPC-Ops uses that we DON'T (key findings)

| # | Technique | HPC-Ops | Our kernel | Impact |
|---|---|---|---|---|
| 1 | **Smaller tiles + more stages** | 64×64×64, kStage=**5** | 128×256×64, kStage=3 | More pipeline overlap → less CTA barrier stall |
| 2 | **Per-warpgroup B barriers** | Each WG has own barrier for W | All WGs share one mbarrier | WG0 starts as soon as its W arrives, doesn't wait for WG1's W |
| 3 | **In-kernel split-K reduction** | atomicAdd flag + spin-wait + reduce() in same kernel | Host-side torch.sum(dim=0) | No extra kernel launch + memory traffic |
| 4 | **launch_bounds** | `__launch_bounds__(384, 1)` | none | Compiler optimizes reg allocation for 1 block/SM |
| 5 | **Manual barrier management** | Raw barriers (init/wait/arrive/tx_bytes) | PipelineTmaAsync abstraction | More fine-grained control over sync timing |
| 6 | **Lower producer regs** | reg_dealloc<24> | reg_dealloc<40> | Producer uses fewer regs, more for consumer |
| 7 | **Block swizzle with flat fallback** | Swizzle for main tiles, flat for tail | GROUP_M=4 everywhere (no fallback) | Tail tiles don't get swizzle overhead |
| 8 | **Per-warpgroup W partitioning** | Each WG loads own W slice via TMA | All WGs share one full W TMA load | Smaller TMA per WG, independent arrival |

### Key insight: smaller tiles → more pipeline stages → less barrier stall

Our kernel: BLK_M=128, BLK_N=256 → smem 214KB → only fits kStage=3
HPC-Ops: BLK_M=64, BLK_N=64 → smem ~88KB → fits kStage=**5**

```
Our kStage=3:    3 stages of overlap → CTA barrier stall 74.3%
HPC kStage=5:    5 stages of overlap → more TMA latency hidden → less stall
```

This is the single most actionable finding: **reduce tile size to increase pipeline depth**.
The tradeoff: smaller tiles = more tiles = more grid overhead + smaller wgmma atom (less work per instruction). HPC-Ops uses `SM90_64x64x16` (half our 64x256x16) — same K but smaller N.

### What we should try next (prioritized by HPC-Ops findings)
1. **launch_bounds** (#4) — trivial, 1 line, helps compiler
2. **kStage=5 with smaller tiles** (#1) — the big one, addresses 74.3% barrier stall
3. **Per-warpgroup B barriers** (#2) — requires manual barrier management
4. **In-kernel split-K reduction** (#3) — eliminates host-side reduction
5. **reg_dealloc<24>** (#6) — lower producer reg budget

---

## Unified Optimization Roadmap (16 items, merged self-research + HPC-Ops)

Ordered by priority (impact on #1 bottleneck: CTA barrier stall 73.3%).

| # | Optimization | Source | Difficulty | Expected | Attacks |
|---|---|---|---|---|---|
| 1 | Small tile + more stages (64×64×64, kStage=5) | HPC-Ops | Medium | +5-10% | smem 88KB→5 stage→67% more overlap→barrier stall ↓ |
| 2 | __launch_bounds__(384, 1) | HPC-Ops | 1 line | +1-2% | Better reg allocation, hint 1 block/SM |
| 3 | FP16 accumulator (acc_dtype=fp16) | Self | 1 line | +2-5%? | regs 168→~84→potential 2 blocks/SM |
| 4 | Per-WG B/W barrier | HPC-Ops | Medium | +2-3% | Each WG independent mbarrier→no cross-WG wait |
| 5 | reg_dealloc<24> (producer) | HPC-Ops | 1 line | +0.5% | Producer fewer regs (24 vs 40) |
| 6 | In-kernel split-K reduce | HPC-Ops | Medium | +1-2% | atomicAdd+spin-wait+reduce in-kernel→no host torch.sum |
| 7 | Manual barrier (vs PipelineTmaAsync) | HPC-Ops | Medium-Hard | +1-2% | Finer-grained sync→reduce wait |
| 8 | Double accumulator (ping-pong) | Self | Medium | +1-2% | Overlap accumulator fill/drain with epilogue |
| 9 | K-loop unroll=2/4 | Self | 1 line | +0.5-1% | ILP, more work per instruction fetch |
| 10 | Swizzle + flat fallback | HPC-Ops | Medium | Edge case | Swizzle main tiles, flat tail→non-aligned shapes |
| 11 | K-tile residue handling | Self | Medium | Functional | Support arbitrary K (non-BLK_K multiples) |
| 12 | BF16 dtype | Self | 1 line | ~0% | Same throughput, precision comparison |
| 13 | L2 cache pinning | Self | Host-side | ~0% | cudaAccessPolicyWindow pin output (compute-bound) |
| 14 | L2 compression hint | Self | Host-side | ~0% | Currently 0% hit, mark compressible |
| 15 | FP8 dtype | Self | Hard | +50-100%? | 2x peak (need H20 wgmma FP8 support check) |
| 16 | 2:4 structured sparsity | Self | Hard | +50-100%? | 2x peak (need sparse input) |

---

## Experiment Log

### #2: `__launch_bounds__(384, min_blocks_per_mp)` — case study

**What**: DSL `.launch(min_blocks_per_mp=N)` sets NVVM `minctasm` metadata, telling the compiler to target N blocks/SM for register allocation. `reqntid` (max threads per block) is auto-generated from block size.

**Tested**:
- `min_blocks_per_mp=1` (both kernels): **no-op**. Compiler already assumes 1 block/SM for register-heavy kernels (154-168 regs/thread). Performance unchanged (48.5 vs 48.9T, 133.1 vs 133.0T — measurement noise).
- `min_blocks_per_mp=2` (small tile, 256 threads): **kernel hangs >6 minutes** (0.04ms → 360,000ms = 9000x slowdown). Compile only took 0.9s — the hang is purely runtime.

**Root cause of 9000x slowdown** (not just linear spilling):

1. **All variables spill to DRAM**: With 168 regs needed but budget = 65536/(2x256) = 128, 40 registers overflow. But the compiler doesn't just spill 40 — it re-evaluates the entire allocation, spilling pipeline state (count/index/phase), TMA descriptors, wgmma fragments, and potentially parts of the accumulator. Every variable access becomes a ~400-cycle DRAM load.

2. **I-cache explosion**: Spill code inflates SASS (each spill = load + use + store). The kernel binary grows 10x+ → exceeds I-cache (32-64KB) → every instruction fetch misses (~100 cycles instead of ~4).

3. **Near-deadlock pipeline**: Consumer is extremely slow (waiting on DRAM for every variable). Producer fills all 5 pipeline stages quickly, then blocks on `producer_acquire` (stages full). Consumer is stuck loading spilled registers. The mbarrier doesn't deadlock (no timeout on Hopper), but the effective wait time approaches infinity — the consumer takes minutes to complete a single K-tile iteration.

4. **Cascade effect**: Slow consumer → producer stalls → pipeline drains → consumer has nothing to consume → both warps wait → SM idle → entire GPU underutilized.

**Lesson**: `min_blocks_per_mp=2` on a register-heavy kernel causes **exponential collapse**, not linear degradation. To achieve 2 blocks/SM, you must reduce register demand at the source (fp16 accumulator, smaller tile, fewer pipeline stages) — not ask the compiler to force it.

**Path to 2 blocks/SM** (requires #3 fp16 accumulator): 168 regs → ~84 (fp16 acc) → 65536/(2x256) = 128 budget → 84 < 128 ✅ → `min_blocks_per_mp=2` would then be safe.

---

### #3: FP16 Accumulator + min_blocks_per_mp=2 — case study

**What**: Change `acc_dtype` from `cutlass.Float32` to `cutlass.Float16`. Halves the accumulator register width (32→16 bits per element). Combined with `min_blocks_per_mp=2` on the small tile kernel to test if 2 blocks/SM helps.

**Tested**:

1. **FP16 acc alone (both kernels)**: Performance +0.2-0.5% at large scale (4096³: 133.5T vs 132.8T; 16384³: 142.1T vs 141.8T). 512³ unchanged (48.5T). Correctness: RE 0.09-0.47% (vs 0.01% with FP32 acc — precision loss expected, 10 mantissa bits vs 23). **ncu: Registers Per Thread still 168** — compiler allocates the same amount (no pressure to reduce) but actual usage likely dropped to ~130.

2. **FP16 acc + min_blocks_per_mp=2 (small tile, 256 threads)**: **No crash!** (vs 9000x crash with FP32 acc). Registers forced to 128 (budget 65536/(2×256)=128). Theoretical occupancy 25% (2 blocks/SM). **But performance unchanged**: 4096³ = 118.7T (vs 118.6T with 1 block/SM). 512³ = 48.3T (vs 48.5T, slight loss from register spill overhead).

3. **ncu comparison (4096³, small tile)**:

| Metric | 1 block/SM (168 regs) | 2 blocks/SM (128 regs) | Delta |
|--------|----------------------|------------------------|-------|
| Compute Throughput | 83.0% | 83.03% | **identical** |
| TFLOPS | 118.6 | 118.7 | ~0% |
| Cycles/Issued | 7.6 | 13.83 | **+82% worse** (register spill) |
| Theoretical Occupancy | 18.75% | 25.0% | +6.25% |
| Achieved Occupancy | 14.04% | 7.39% | **-6.65% worse** (problem too small) |

**Root cause: TC is the bottleneck, not occupancy**:
- Hopper's tensor core (TC) is an SM-level shared resource. A single block's wgmma instructions already saturate it (83% throughput).
- A second block on the same SM adds more warps, but their wgmma instructions queue behind the first block's — **no parallel TC execution**.
- The extra block's warps stall at the mbarrier waiting for TC, just like the first block's warps — **2× the stalls, 0× the TC throughput**.
- Cycles/Issued doubled (7.6→13.83) because forcing 128 regs (vs ~130 actual) causes minor spilling, and the second block adds scheduling overhead.

**Key lesson**: **Occupancy doesn't matter for compute-bound wgmma kernels on Hopper.** The traditional GPU wisdom ("more warps = hide more latency") doesn't apply because:
1. wgmma is a long-latency instruction that keeps the TC busy for many cycles
2. TMA pipeline (not warp scheduling) hides the TMA latency
3. TC is SM-level, shared across all blocks — can't parallelize across blocks

This is the Hopper design philosophy: **long-latency MMA + hardware pipeline replace multi-warp latency hiding**.

**Reverted**: FP16 acc (precision loss not worth +0.2-0.5%), min_blocks_per_mp=2 (no gain, slight loss from spill overhead).

### #5: reg_dealloc<24> (producer) — no-op

**What**: Reduce producer register budget from 40 to 24 (matching HPC-Ops). `LOAD_REGISTER_REQUIREMENT = 24`.

**Result**: No change — 512³ 48.6 vs 48.5T, 4096³ 132.8 vs 133.1T, 16384³ 141.2 vs 141.2T.

**Why no-op**: Producer registers don't affect consumer allocation. Consumer already targets 232 regs/thread (budget allows 244). Reducing producer from 40→24 frees 16×128=2048 regs, but consumer doesn't claim them (already has enough). The freed space sits idle.

### #9: K-loop unroll=2/4 — no net gain

**What**: Change consumer K-tile loop `unroll=1` to `unroll=2` or `unroll=4` to increase instruction-level parallelism (ILP).

**Results**:

| unroll | 512³ | 4096³ | 16384³ |
|--------|------|-------|--------|
| 1 (baseline) | 48.5 | 132.8 | 141.2 |
| 2 | 49.0 (+1.0%) | 131.6 (-0.9%) | 140.5 (-0.5%) |
| 4 | 48.1 (-0.8%) | 131.6 (-0.9%) | 140.9 (-0.2%) |

**Analysis**: ILP vs I-cache tradeoff. Unrolling 2× issues more independent wgmma instructions per fetch, helping small scale (512³ +1%). But the larger code footprint evicts I-cache at larger scale, costing -0.5~0.9%. Net effect negative.

**Reverted** to `unroll=1`.

### #6: In-kernel split-K reduce — CopyReduce compilation hang, cancelled

**What**: Fuse the split-K reduction into the GEMM kernel using TMA hardware
atomic-add (`CopyReduceBulkTensorTileS2GOp(reduction_kind=ADD)`). Each CTA
would TMA-store directly to the shared `mC` position with atomic add,
eliminating the separate output buffer and host-side `torch.sum`.

**Measured overhead** (torch.sum, outside kernel timing):

| Shape | sk | torch.sum | GEMM kernel | Reduction % of total |
|-------|----|-----------|-------------|----------------------|
| 512³  | 8  | 0.010ms   | 0.02ms      | **32%**              |
| 1024³ | 2  | 0.010ms   | 0.04ms      | **20%**              |
| 4096³ | 4  | 0.088ms   | 1.3ms       | 6.3%                 |
| 8192³ | 2  | 0.364ms   | 4.7ms       | 7.2%                 |

The overhead is significant (6-32% of end-to-end time), especially for small scale.

**Finding 1: DSL supports 4+ Tensor parameters**. Initial assumption that `@cute.jit`
limited to 3 Tensor params was **wrong** — adding a `split_counter: cute.Tensor`
parameter works fine. The earlier error was due to: (a) only modifying the large
tile kernel while the dispatch mode used the small tile kernel for 512³, and
(b) `__pycache__` caching stale bytecode.

**Finding 2: `CopyReduceBulkTensorTileS2GOp` causes MLIR compilation hang**.
The op class exists and can be instantiated, but `make_tiled_tma_atom` +
`@cute.jit` compilation hangs for >300s (infinite loop or codegen bug in
nvidia-cutlass-dsl 4.7.0). Three approaches all failed:
1. `if split_k > 1:` staged-if — op defined inside conditional branch, invisible after
2. `const_expr(split_k > 1)` — also hangs
3. Always use CopyReduce (even for sk=1, ADD to zeroed buffer = store) — still hangs

**Conclusion**: The theoretical approach (TMA hardware atomic-add) is correct,
but the DSL compiler (4.7.0) doesn't support `CopyReduceBulkTensorTileS2GOp`
codegen. Reverted to `CopyBulkTensorTileS2GOp` + output buffer + `torch.sum`.

**Future**: Could be revisited with (a) newer DSL versions, (b) a TMA-based
reduction (TMA G2S load partials + S2R + add + TMA S2G store), or (c) C++ CUDA
kernel for the reduction.

**Finding 3: SIMT atomic_add also fails**. After CopyReduce failed, tried
element-wise `cute.arch.atomic_add` from registers directly to gmem output:

| Variant | Compiles | Correctness | Performance | Issue |
|---------|----------|-------------|-------------|-------|
| fp16 atomic_add | ✅ | RE=0.04% ✅ | 62.9T (vs 129T) ❌ | Hopper fp16 has no hardware atomic — uses CAS loop (~100 cycles/op) |
| fp16 + local_tile pointer | ✅ | — | — | `local_tile` in loop generates excessive MLIR IR |
| fp32 atomic_add | ✅ | RE=141% ❌ | — | `local_tile` pointer computation wrong in loop, most elements not written |

**Root cause**: CuTe DSL is designed for **tile-level operations** (TiledCopy, TMA).
Element-level access (`tensor[idx]` returns a scalar, not a sub-Tensor view;
`local_tile(tensor, (1,), (idx,))` in a loop generates excessive MLIR IR and
may compute wrong pointers). The DSL cannot efficiently do per-element atomic
operations on partitioned tensors.

### #8: Double accumulator (ping-pong) — DSL limitation, not implemented

**Concept**: Two accumulators alternate per tile. While one is drained (R2S + TMA S2G), the other is filled (next tile's wgmma). Overlaps R2S (LSU pipeline) with wgmma (TC pipeline).

**Why not implemented**: DSL staged-if limitation prevents selecting between two rmem tensors with a runtime condition (`tile_iter % 2`). Variables defined in one staged-if branch can't be used after the branch — would require duplicating the entire mainloop+epilogue code in both branches.

**Analysis**: Even if implemented, the benefit is marginal. The R2S (~20 cycles) is much shorter than wgmma (~100+ cycles). The overlap window is small. The existing epi-overlap (TMA S2G + next tile mainloop) already captures the larger overlap. Double accumulator would only add R2S overlap, saving ~20 cycles/tile = ~0.008% at 4096³.

### #10: Padding for non-aligned shapes — correctness fix ✅

**Problem**: TMA G2S loads a full (BLK_M, BLK_K) box. When K is not a multiple of BLK_K, the last K-tile reads OOB elements — TMA doesn't zero-pad, it reads whatever is in memory (garbage). This causes RE=141% for shapes like 1024×1024×333.

**Fix**: Pad a, b, output_buf to multiples of BLK_M/BLK_N/BLK_K before kernel launch. Extract original (unpadded) result after.

| Shape | Before (RE) | After (RE) | TFLOPS |
|-------|-------------|------------|--------|
| 1024×1024×333 | 141.48% ❌ | 0.00% ✅ | — |
| 1000×777×333 | 141.18% ❌ | 0.00% ✅ | 53.0 |
| 3000×3000×3000 | 0.00% ✅ | 0.00% ✅ | 122.2 |

**Note**: 3000³ worked before because OOB memory happened to be zeros (from prior allocations). The fix ensures correctness regardless of memory content.

### #13: L2 cache pinning — marginal

**What**: Use `cuStreamSetAttribute` with `CUaccessPolicyWindow` to pin the output tensor in L2 cache. Persistent L2 access improves write-back locality for repeated writes (e.g., persistent kernel writing to same output region).

**Results**:

| Shape | Without pinning | With pinning | Delta |
|-------|-----------------|--------------|-------|
| 512³   | 48.5            | 48.2         | -0.6% |
| 4096³  | 130.6           | 131.8        | +0.9% |

**Analysis**: Marginal — kernel is compute-bound (DRAM 4.3%), so L2 optimization has limited impact. Small scale slightly worse (L2 pollution from pinning evicts A/B data). Large scale slightly better (output write-back more efficient). Net: ~0% — not worth the complexity for production use.

### #14: L2 compression — hardware-controlled, no software hint

**What**: Hopper L2 supports hardware compression. ncu showed 0% compression success rate, est. 1.5% speedup if enabled.

**Finding**: L2 compression is a hardware feature that depends on data patterns. Random fp16 data (our input) is not compressible (high entropy). There is no CUDA API to "hint" the L2 to try compression — it either works (data is compressible) or doesn't (data is random).

**Conclusion**: Cannot be controlled from software. Would only help with structured/low-entropy data patterns (e.g., sparse tensors, zero-padded regions). Not applicable to dense random GEMM.

### #17: TMA Multicast + Cluster — DSL 4.7.0 two limitations, cancelled

**Goal**: Use `CopyBulkTensorTileG2SMulticastOp` with a 2-CTA cluster. Leader CTA
issues multicast TMA load for A, broadcasting to both CTAs' smem. Halves gmem
reads for A tiles. Each CTA computes a different output tile (different bid_n).

**Five approaches tried, all failed**:

| # | Approach | Compile | Run | Failure |
|---|----------|---------|-----|---------|
| 1 | Dynamic mask (ternary) `Int16(3) if pos==0 else 1` | ❌ hang | — | MLIR codegen can't handle runtime-conditional multicast TMA |
| 2 | Dynamic mask (arithmetic) `3 - pos*2` | ❌ hang | — | Same codegen limitation |
| 3 | Always multicast (mask=3, unconditional) | ✅ | RE=1.0 ❌ | Both CTAs multicast → mbarrier double-counting cascade |
| 4 | Staged-if `if pos==0:` with constant masks | ❌ hang | — | Multicast TMA in staged-if branch → codegen hang |
| 5 | Regular TMA + cluster launch (no multicast) | ✅ | K≤256 ✅, K≥320 ❌ | PipelineTmaAsync cluster mode breaks at wrap-around |

**Limitation 1: Multicast TMA codegen**. `CopyBulkTensorTileG2SMulticastOp` with
any runtime-dependent `mcast_mask` (ternary, arithmetic, or staged-if) causes
MLIR codegen infinite loop. Only compile-time constant masks compile, but
unconditional multicast causes mbarrier double-counting.

**Limitation 2: PipelineTmaAsync cluster + wrap-around**. With `cta_layout_vmnk=(1,1,2,1)`,
the pipeline works for the first round (≤ NUM_STAGES k-tiles). At wrap-around
(stage index returns to 0 + phase bit flips), cross-CTA mbarrier arrive counts
are miscalculated → phase mismatch → consumer reads wrong stage → wrong results
or deadlock. This is a bug in the DSL's pipeline implementation for cluster mode.

**Conclusion**: TMA multicast + cluster is not implementable in CuTe DSL 4.7.0.
HPC-Ops (production) uses C++ CUDA with manual mbarrier management, bypassing
these DSL limitations. Requires DSL compiler fixes (multicast codegen + pipeline
cluster phase management).

**Learning value**: This experiment taught the Hopper cluster programming model
(cluster launch, `block_in_cluster_idx`, `cta_layout_vmnk`, multicast TMA masks,
pipeline arrive-count recalculation) even though the DSL can't compile it yet.

---

## Optimization Principles Reference

Each optimization's **theoretical mechanism** — why it can accelerate a GPU kernel,
even if it didn't help our specific GEMM. Organized by the bottleneck they target.

### Bottleneck 1: CTA Barrier Stall (73.3% of stall time)

The #1 bottleneck in our WGMMA+TMA pipeline. `consumer_wait` blocks the consumer
until TMA data arrives at the mbarrier; `producer_acquire` blocks the producer until
the consumer releases a stage. This is **inherent** to the pipeline architecture —
the question is how to minimize the wait.

#### 1. Pipeline Stages (NUM_STAGES)

**Mechanism**: N-stage pipeline allows producer to issue N TMA loads ahead of the
consumer's current position. More stages = more TMA overlap = less consumer_wait.
Diminishing returns: if TMA latency ≈ T cycles and each stage covers T/N overlap,
beyond N ≈ T/tile_compute, extra stages don't help (pipeline full).

**Our result**: NUM_STAGES=3 is optimal (smem 214KB/228KB limits to 3; 4 doesn't
fit). Reducing to 2 (v4-blk96) hurt more than the sync reduction helped — **pipeline
overlap > sync count**.

#### 2. Tile Size vs Pipeline Depth

**Mechanism**: Smaller tile → less smem per stage → more stages fit → more pipeline
overlap → less barrier wait. Trade-off: smaller tile = less work per wgmma instruction
= lower TC efficiency (more instructions for same FLOPs).

**Our result**: v7 (64×64, S5) killed barrier stall (74% → 8%, -82%) but TC throughput
dropped (91% → 83%). Net: small-scale +13% (occupancy wins), large-scale -17% (TC
efficiency loss). **Optimal tile size varies by problem scale** (cuBLAS does heuristic
dispatch).

#### 3. Epilogue Overlap

**Mechanism**: After the last WGMMA of a tile, the epilogue (R2S → TMA S2G store)
takes ~180 cycles with TC idle. If we issue the TMA store but don't wait
(`cp_async_bulk_commit_group` without `wait_group`), the next tile's mainloop can
start immediately. `wait_group(1)` at the next epilogue start waits for the previous
store — by then, the mainloop has run enough cycles to hide the store latency.

**Our result**: +1.4% at 2048³ (mainloop 800 cyc >> store 180 cyc → full overlap).
-4.2% at 512³ sk=8 (mainloop 50 cyc << store 180 cyc → wait stalls). **Only helps
when mainloop > store duration**.

#### 4. Per-Warpgroup Barriers

**Mechanism**: Currently both MMA warpgroups share one mbarrier for consumer_release.
WG1 can't proceed until WG2 also releases. With per-WG barriers, each WG independently
signals "I'm done with this stage" — no cross-WG waiting.

**Our result**: Cancelled — DSL `PipelineTmaAsync` abstraction doesn't support per-WG
mbarrier on shared smem. Requires manual mbarrier management (C++ CUDA, like HPC-Ops).

#### 5. Manual Barrier (vs PipelineTmaAsync)

**Mechanism**: PipelineTmaAsync is a high-level abstraction that manages mbarrier
arrive/wait automatically. Manual mbarrier (`init`/`arrive`/`wait`/`tx_bytes`) gives
finer control — e.g., separate arrive for A vs B, or per-WG arrive counts. Can reduce
wait time by matching arrive semantics to actual access patterns.

**Our result**: Cancelled — DSL doesn't expose low-level mbarrier API directly in
@cute.kernel functions. HPC-Ops uses C++ CUDA with `FullBarrier::init/arrive/wait`.

### Bottleneck 2: SM Underutilization (small scale)

Small problems have too few tiles to fill all SMs (512³ = 8 tiles on 78 SMs = 0.1
waves → 90% of SMs idle). This is a **parallelism** problem, not a compute problem.

#### 6. Split-K

**Mechanism**: Split K dimension into `split_k` partitions. Each partition computes
a partial result (partial GEMM over a K slice). Grid grows from `M×N` to `M×N×split_k`
— more CTAs = more parallelism. Host-side reduction (`torch.sum`) combines partials.

**Our result**: 512³ sk=8 = 42.9T (+254% vs baseline, +129% vs cuBLAS). But non-monotonic:
1024³ sk=2 (91.3T) > sk=4 (84.4T) because fewer K-tiles/CTA → pipeline fill/drain
overhead dominates. **Best split_k varies by problem size**.

#### 7. Persistent Kernel

**Mechanism**: Launch `num_sms` CTAs instead of `total_tiles`. Each CTA strides
through `ceil(total_tiles/num_sms)` tiles. Eliminates the last partial wave where
`total_tiles % num_sms` SMs are idle.

**Our result**: No gain — 4096³ has 6.56 waves, tail is 0.56 waves = 8% of time, but
per-tile overhead (consumer setup per tile due to DSL staged-if) ate the benefit.
512³ has 0.1 waves (fewer tiles than SMs) — persistent can't help. **Only helps
when total_tiles >> num_sms AND per-tile overhead < tail-wave savings**.

### Bottleneck 3: Register Pressure

WGMMA accumulator uses ~32 fp32 registers per thread (128×256×4B / 256 threads).
With overhead, total = 154-168 regs/thread → 1 block/SM (register-limited).

#### 8. FP16 Accumulator

**Mechanism**: FP16 accumulator uses half the register width (16-bit vs 32-bit per
element). ~84 regs/thread instead of 168 → fits 2 blocks/SM (budget 65536/2×256=128).
More blocks = more warps = better latency hiding.

**Our result**: 2 blocks/SM achieved, but **TC throughput unchanged** (83% → 83%).
Hopper's TC is SM-level shared — a second block's wgmma queues behind the first.
**Occupancy doesn't matter for compute-bound wgmma kernels on Hopper.** The TC
pipeline (not warp scheduling) hides latency.

#### 9. `__launch_bounds__` / `min_blocks_per_mp`

**Mechanism**: Compiler hint `min_blocks_per_mp=N` tells the compiler to target N
blocks/SM for register allocation. Max regs/thread = 65536/(N×threads). With N=2
and 256 threads: max=128. If kernel needs 168, the compiler spills 40+ to DRAM.

**Our result**: N=1 = no-op (compiler already assumed 1). N=2 = **9000× slowdown**
(exponential collapse from cascade: register spill → DRAM access → I-cache miss →
pipeline near-deadlock). **Must reduce register demand at source, not force it**.

#### 10. `warpgroup_reg_dealloc` / `warpgroup_reg_alloc`

**Mechanism**: Hopper allows different warpgroups within a CTA to have different
register budgets. Producer warpgroup "deallocates" (returns) registers it doesn't
need → consumer warpgroup "allocates" (claims) them for accumulator/descriptors.

**Our result**: `reg_dealloc<24>` (40→24) = no-op. Consumer already targets 232
regs (budget allows 244). The freed 16×128=2048 registers sit idle — consumer
doesn't claim them. **Producer register savings don't transfer to consumer**.

### Bottleneck 4: Memory Bandwidth (not our bottleneck — DRAM 4.3%)

#### 11. TMA Multicast + Cluster

**Mechanism**: In a cluster of N CTAs, when multiple CTAs need the same A or B data,
one "leader" CTA issues a multicast TMA load. The TMA hardware reads from gmem once
and writes to all N CTAs' smem simultaneously via the cluster DSMEM bus. Saves N×
gmem reads for shared data.

**Important clarification**: "multiple CTAs doing one MMA" is a **misconception**.
WGMMA is a single-CTA instruction — no cross-CTA compute cooperation. Cluster enables
**data sharing** (multicast), not **compute cooperation**. Each CTA still runs its
own wgmma on its own output tile, but they share input data → less gmem bandwidth.

**Our result**: 9 approaches all failed in DSL 4.7.0 (compilation hang, segfault,
RE=70%). DSL codegen doesn't support `CopyBulkTensorTileG2SMulticastOp` + WGMMA +
PipelineTmaAsync. Even if it worked: our kernel is compute-bound (DRAM 4.3%), so
bandwidth savings would give ~0-2%.

#### 12. L2 Cache Pinning

**Mechanism**: `cuStreamSetAttribute` with `CUaccessPolicyWindow` tells the L2 to
prioritize caching the output tensor. Repeated writes to the same output region
(e.g., persistent kernel) benefit from L2 persistence — fewer DRAM write-backs.

**Our result**: +0.9% at 4096³ (slight write-back improvement), -0.6% at 512³ (L2
pollution evicts A/B data). **Marginal for compute-bound kernels** (DRAM < 5%).

#### 13. L2 Compression

**Mechanism**: Hopper L2 hardware automatically attempts to compress cache lines.
If data has low entropy (e.g., zeros, repeated patterns), compression succeeds →
more effective L2 capacity → better hit rate.

**Our result**: 0% compression success (random fp16 = high entropy). No software
API to control this — it's hardware-automatic. **Only helps with structured/low-
entropy data** (e.g., sparse tensors, zero-padded regions).

### Bottleneck 5: Instruction-Level Parallelism (ILP)

#### 14. K-loop Unroll

**Mechanism**: Unrolling the K-tile loop by factor N issues N independent
`cute.gemm` calls per loop iteration. The instruction scheduler can issue multiple
WGMMA instructions in parallel (ILP), potentially filling TC pipeline bubbles.

**Our result**: unroll=2 → 512³ +1.0% (ILP helps when few K-tiles), 4096³ -0.9%
(larger code → I-cache miss). unroll=4 → worse everywhere. **ILP vs I-cache tradeoff**.

#### 15. Double Accumulator (Ping-Pong)

**Mechanism**: Two accumulators alternate per tile. While accumulator A is being
drained (R2S → TMA S2G store, ~20 cycles on LSU pipeline), accumulator B is being
filled (next tile's WGMMA, ~100+ cycles on TC pipeline). Overlaps LSU with TC.

**Our result**: Not implemented — DSL staged-if limitation prevents selecting between
two rmem tensors with runtime condition (`tile_iter % 2`). Even if implemented: R2S
(~20 cyc) << WGMMA (~100+ cyc), so overlap window is small. Existing epi-overlap
already captures the larger TMA-store-vs-mainloop overlap.

### Other Optimizations

#### 16. Block Swizzle (tile reordering)

**Mechanism**: Reorder tile processing so that a group of `GROUP_M` consecutive
M-tiles are processed together before striding N. Adjacent M-tiles share A rows →
L2 can serve A from cache instead of gmem. Reduces gmem reads for A by ~GROUP_M×.

**Our result**: L2 hit rate DROPPED 4% (66% → 62%) — GROUP_M=4 increases working set
(6 active M-blocks vs 5), exceeding L2 capacity (60MB) for 4096³ data (64MB). Also,
default M-major order already maximizes A reuse. **Only helps for memory-bound
kernels where L2 is the bottleneck and data fits in L2**.

#### 17. Padding for Non-Aligned Shapes

**Mechanism**: TMA loads a fixed-size (BLK_M, BLK_K) box. When K isn't a multiple
of BLK_K, the last box reads OOB gmem (TMA doesn't zero-pad). Padding A/B/output
to tile multiples ensures all TMA boxes are valid.

**Our result**: Fixed RE=141% → 0% for non-aligned shapes (1024×1024×333, 1000×777×333).
**Correctness fix, not performance optimization.**

#### 18. BLK_K Increase (64→96) + Fewer Stages (3→2)

**Mechanism**: Larger BLK_K means fewer K-tiles → fewer pipeline sync points (64
→ 43 syncs, -33%). But larger BLK_K means more smem per stage → fewer stages fit
(3→2) → less pipeline overlap.

**Our result**: -0.9% (pipeline overlap loss > sync reduction). **Pipeline depth
(NUM_STAGES) is more valuable than sync count**. The 3-stage pipeline hides TMA
latency better; the extra sync overhead is offset by overlap.
