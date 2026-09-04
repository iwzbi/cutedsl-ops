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

*(v1-v6 = FP8 big-kernel line. The fp16 cluster line — 94.1 T @1024³, and its
`gemm-v9-multicast` hardware-multicast variant, neutral on this compute-bound
GPU — is tracked in case studies #17/#18 and the cluster-kernel entries below.)*

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

### #17: TMA Multicast + Cluster — cancelled at the time (both 'limitations' later dissolved; see #18)

> **[RETRADED 2026-09-03 — see #18 below.]** Multicast IS implementable in DSL
> 4.7.0: the missing piece was the lesson-12 recipe (`CopyBulkTensorTileG2SMulti
> castOp` + `tma_partition` on the shared axis + static `make_layout_image_mask`
> + cluster-aware pipeline arrive counts). The historical RE=70.56% failure
> reproduces exactly from one wrong `slice_` axis (double full-tile issue), which
> is what likely killed the original attempt. Measured outcome on H20: mechanism
> verified in SASS (`UTMALDG.2D.MULTICAST`), performance neutral — GEMM here is
> compute-bound, so the halved L2 reads have nothing to relieve.

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

**Conclusion (as of the experiment — SINCE RETRACTED, see #18)**: ~~TMA
multicast + cluster is not implementable in CuTe DSL 4.7.0. Requires DSL
compiler fixes.~~ #18 implemented it in the same DSL 4.7.0: 'Limitation 1' was
self-inflicted (wrong `slice_` axis -> double full-tile issue, RE=70.56%
reproducible), and 'Limitation 2' (wrap-around) is handled by `defer_sync=True`
+ explicit `pipeline_init_arrive/wait`.

**Learning value**: This experiment taught the Hopper cluster programming model
(cluster launch, `block_in_cluster_idx`, `cta_layout_vmnk`, multicast TMA masks,
pipeline arrive-count recalculation) even though the DSL can't compile it yet.

---

### #18: TMA Multicast RESOLVED — real hardware multicast lands; neutral on H20

**Verdict: works, shipped (`gemm-v9-multicast`), gains nothing measurable on
this GPU — and the 'why nothing' is itself the finding.**

**Method recovered from cutlass-notes lesson 12** (patched 2 `fence_proxy
(ProxyKind…)` call sites to the string-literal API — the lesson's only staleness;
it then passes its full 640-case sweep incl K=8192). Recipe ported onto the
existing cluster kernel (CLUSTER_M=2 shares B across 2 CTAs stacked along grid-y):
1. host: `make_tiled_tma_atom(CopyBulkTensorTileG2SMulticastOp(), mB, …,
   num_multicast=CLUSTER_M)` — note `G2SOp + num_multicast=2` is rejected at
   compile time, so a successful build already proves the multicast op.
2. device: `tma_partition(b_atom, cta_coord=cluster_coord_mnk[0],
   cta_layout=make_layout(slice_(cta_layout_mnk,(None,0,0)).shape), …)` — each
   CTA issues its 1/2 slice of the shared B tile; `b_mcast_mask =
   make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)` (static);
   copy carries `mcast_mask=b_mcast_mask`.
3. pipeline: consumer arrive count × `(CLUSTER_M+CLUSTER_N-1)` with the true
   `cta_layout_vmnk=(1,2,1,1)`; non-multicast side stays (1,1,1,1)-style plain.
4. lifetime: `pipeline_init_arrive(cluster_shape_mn=(2,1), relaxed)` + wait.

**The bug that had killed it (#17's RE=70.56% reproduced exactly)**: copying
lesson-12's *B*-branch `slice_(cta_layout_mnk,(0,None,0))` when our shared axis
is **M** — both CTAs then request the full tile and the pair double-issues,
whose signature is numerically-correct-but-slower OR ~70% error variants.
Correct branch shape: `(None,0,0)` + `cluster_coord_mnk[0]` (lesson-12's A-case).

**Evidence chain (idle H20, same-session A/B against the no-mcast cluster
kernel of the same tree):**
| shape | no-mcast | mcast |
|---|---|---|
| 1024³  | 94.1 T | 93.6 T |
| 4096³  | 129.7 T | 129.4-129.6 T |
| 8192²×256 | 102.8 T | 102.5-102.6 T |
| ncu @4096³ | dram rd 254.7 MB, lts 1.53 GB | 249.9 MB, 1.51 GB |
SASS: `UTMALDG.2D` (A) + `UTMALDG.2D.MULTICAST` (B) — hardware multicast
genuinely executes. Correctness: 4/4 shapes (incl 8192²×256 deep-K wraparound),
RE ≤ 0.01%.

**Why no gain (the actual Hopper-on-H20 lesson)**: every compute-bound shape here
runs at 2-3% DRAM utilization; the L2 read redundancy multicast removes was never
the limiter. Multicast pays off on bandwidth-hungry parts (H100/B200 SXM, real
LLM GEMM N≫M, or *multi-tenant* GPUs where L2/DRAM contention is external —
H20-with-co-tenant is a poor man's version of that). Also lts__t_bytes barely
moved: with each CTA issuing its own 1/2 slice, L2 still answers the same total
sector count; multicast's win is DRAM-side fetch dedup + xbar traffic, which a
60 MB-L2-resident B never exercises. Kept for (a) mechanism availability,
(b) robustness under co-tenancy, (c) future bandwidth-bound shapes.

### #19: Bidirectional multicast — A-side, B-side, and the 2×2 rank/coord trap

**Verdict: bidirectional support ships (`gemm-v10-bimcast`, default `(2,1)`).
All three configs now correct — the 2×2 rank/coord mismatch was a CuTe
column-major-vs-hardware-x-fastest stride trap, not a DSL limit — but 2×2
multicast measures **slower** (−25~46%) from cross-CTA stage-release cost, so
the practical win envelope is one-directional, 2-CTA stripes.**

Extends #18 from "B multicast along M" to the full 2-D scheme (principle
diagram in section 12). Four edits on #18's skeleton (which already computed
both masks — only A-side plumbing was missing): A's host op becomes
`G2SMulticastOp` when `CLUSTER_N>1`, A's atom gains `num_multicast=CLUSTER_N`,
A gets the mirrored `tma_partition` branch (`slice_ (0,None,0)`, coord
`cluster_coord_mnk[1]`), and both A producer copies take `mcast_mask=a_mcast_mask`.

**The 2×2 failure.** `(2,1)` and `(1,2)` passed immediately; `(2,2)` gave
RE=122.5% and stayed broken even with either operand reverted to plain loads
(E2/E3) — so it was the cluster geometry, not the new A code. Root cause:
`make_layout((CLUSTER_M, CLUSTER_N, 1))` defaults to column-major (**m
fastest**) while hardware cluster ranks are **grid-x = n fastest**; degenerate
axes mask the swap, a true 2×2 exposes it (every member computes the wrong
slice coord and mask). Fix = pinned strides:

```python
cta_layout_mnk  = cute.make_layout((CLUSTER_M, CLUSTER_N, 1), stride=(CLUSTER_N, 1, 1))
cta_layout_vmnk = cute.make_layout((1, CLUSTER_M, CLUSTER_N, 1),
                                   stride=(CLUSTER_M * CLUSTER_N, CLUSTER_N, 1, 1))
```

(`enable_multicast_signaling=True`, E1, reproduced the identical error — the
hand-rolled `(CLUSTER_M+CLUSTER_N-1)*NUM_WARPS` arrive count was never the issue.)

**Results** (post-fix; correctness 9/9 = 3 configs × {1024³, 4096³,
8192²×256} all Success, RE ≤ 0.01%; TFLOPS on idle GPU, same session):

| config | multicast | 1024³ | 4096³ | 8192²×256 | vs (2,1) |
|---|---|---|---|---|---|
| (2,1) | B only | 93.9 | 130.8 | 103.5 | — |
| (1,2) | A only | 95.0 | 130.4 | 102.7 | ≈ neutral ✓ |
| (2,2) | A+B | **50.9** | **101.8** | **77.9** | **−25~46%** |

New, sharper than "neutral": **one-directional multicast is free in both
directions** — A-side (1,2) matches B-side (2,1) everywhere, confirming the
mechanism is symmetric. But **2×2 is genuinely slow**, and the why is
structural, not a bug: multicast forces the *stage-release* to be cross-CTA —
a member may not recycle a shared tile until **every** peer's warps have
consumed their fan-out copies, so each release becomes DSMEM arrive traffic
sized by `mcast_size` (3 for 2×2 vs 2 for a stripe). At 4 members the release
chain overtakes the wgmma work it was supposed to hide: −46% at 1024³. The
cluster's co-residency/gang constraints likely add on top (2 warpgroups/CTA ×
4 CTAs pinned to one GPC). Not a race — every number is bit-correct.

**Takeaway for sizing**: multicast pays where it dedups an operand *and* the
cluster stripe stays 2 CTAs wide; bidirectional dedup at 2×2 converts a
bandwidth idea into a synchronization tax.

---


## Optimization Principles (Detailed)

### 1. Pipeline Stages (NUM_STAGES)

`NUM_STAGES` is the **software‑pipelining depth** of a GEMM loop: the number of distinct shared‑memory buffer slots used to keep multiple async tile‑loads (`cp.async` on Ampere, **TMA** on Hopper) in flight behind in‑flight `WGMMA` compute. It is the single most important knob for turning a load‑bound kernel into a compute‑bound one — and the most expensive one in shared memory.

---

#### 1. The problem it solves

A GEMM kernel's inner loop is, at the hardware level, two alternating phases per K‑tile:

| phase | instruction | latency (H20, FP16) |
|---|---|---|
| load DRAM → smem | `TMA_LOAD` | **~200 cyc** |
| compute smem → TC | `WGMMA` (M=128,N=256,K=64) | **~100 cyc** |

Run serially (S = 1, single buffer), each K‑iteration costs `200 + 100 = 300` cycles and the tensor cores sit idle for 200 of them:

```
S=1  iter:  |====TMA 200====|==WGMMA 100==|====TMA 200====|==WGMMA 100==|
              TC idle ▲▲▲▲▲▲▲▲▲▲                TC idle ▲▲▲▲▲▲▲▲▲▲
```

TC utilization ceiling here is `100 / 300 = 33 %`. The fix is to **decouple** the two phases: while iteration *k* computes, iteration *k+1, k+2, …* loads. Each in‑flight load needs its **own smem buffer** (you cannot overwrite a buffer still being read by WGMMA). That count of independent buffers is `NUM_STAGES`.

---

#### 2. The latency‑hiding inequality

Steady‑state compute‑bound requires that the compute time per iteration be large enough to **cover** the load latency:

```
NUM_STAGES * T_compute  ≥  T_load
```

For the H20 numbers above (`T_load = 200`, `T_compute = 100`):

| NUM_STAGES | `S * T_compute` | covers 200? | regime |
|---|---|---|---|
| 1 | 100 | ❌ (deficit 100) | load‑bound, TC ≤ 33 % |
| 2 | 200 | ⚠️ exact | boundary — any jitter stalls |
| 3 | 300 | ✅ (slack 100) | compute‑bound, TC → 100 % |
| 4 | 400 | ✅ (slack 200) | compute‑bound, slack wasted |

The threshold is **`S* = ⌈T_load / T_compute⌉ = ⌈200/100⌉ = 2`**, and you want one extra stage of slack → **S = 3 is the sweet spot** for this kernel. Going to S = 4 doubles the smem cost of the extra stage without raising TC utilization — pure waste unless the load latency itself grows (e.g. DRAM contention).

---

#### 3. Timeline: S = 3 in steady state

`K = 4096, BLK_K = 64 → 64 K‑tiles`. Prologue issues 3 loads ahead; then each iteration issues one load and consumes one.

```
t→      0       100      200      300      400      500      600      700  (cyc)
load 0  |=======TMA=======|
load 1          |=======TMA=======|
load 2                   |=======TMA=======|
load 3                            |=======TMA=======|        ← issued while W1 runs
load 4                                     |=======TMA=======|
wgmma0                          |=W=100=|          ← consumes buf0
wgmma1                                   |=W=100=|  ← consumes buf1
wgmma2                                            |=W=100=|  ← consumes buf2
                                  ▲
                          steady state: every 100 cyc a WGMMA retires,
                          TMA completely hidden behind it.
```

Prologue cost = `(S−1) * T_compute` to fill ≈ 200 cyc amortized over 64 tiles → <0.5 % overhead. Throughput in steady state = `1 WGMMA / 100 cyc` → TC utilization ≈ 100 %, matching the measured **90.9 %**.

---

#### 4. The shared‑memory cost (the catch)

Each stage owns a private copy of the A and B tiles it is loading into:

```
smem_per_stage = BLK_M * BLK_K * bytes + BLK_N * BLK_K * bytes
```

For FP16 (2 B), `BLK_M=128, BLK_N=256, BLK_K=64`:

```
A_tile = 128 * 64 * 2 = 16 384 B = 16 KiB
B_tile = 256 * 64 * 2 = 32 768 B = 32 KiB
smem_per_stage = 48 KiB
```

Total smem budget for the kernel:

```
smem_total = NUM_STAGES * smem_per_stage + smem_epilogue
```

With S = 3 and a 64 KiB epilogue (C‑tile accumulation in smem before final r2g):

```
smem_total = 3 * 48 + 64 = 208 KiB   (~214 KiB with padding/fragments)
```

H20 per‑SM shared memory ceiling is **228 KiB** — so S = 3 fits with only ~14 KiB headroom. S = 4 would need `4*48 + 64 = 256 KiB` → **does not fit**, would force shrinking `BLK_M`/`BLK_N` and slash arithmetic intensity. This is why the chosen config stops at 3.

---

#### 5. Reading the ncu profile

| metric | value | interpretation |
|---|---|---|
| **TC utilization** | 90.9 % | WGMMA pipeline nearly saturated — S=3 is doing its job |
| **DRAM utilization** | 7.2 % | **not** bandwidth‑bound; loads are tiny and TMA‑prefetched |
| **barrier stall** | 74.3 % | dominant stall reason |

The **barrier stall = 74.3 %** is *not* "74 % of time wasted" — it is "of the cycles threads spent stalled, 74 % were at a barrier". With TC at 90.9 % that is the **expected healthy signature** of a pipelined kernel: threads finish their WGMMA, walk into `mbarrier.try_wait` for the next tile, and sleep precisely there until the next TMA completes. That is the *right* place to wait. The kernel is compute‑bound, not barrier‑bound.

A red flag would be: TC ≤ 50 % **and** barrier stall dominant → pipeline depth too small, threads idle on a load that hasn't returned. TC 90.9 % rules that out here.

---

#### 6. Pipeline depth vs sync count — the subtle failure mode

There are **two different "depth" notions** that get conflated:

- **Pipeline depth** = `NUM_STAGES` = number of smem buffer slots.
- **Sync count** = number of outstanding async‑ops a thread actually `wait_group`s on at each barrier.

> **Pipeline depth > sync count** → you have allocated more buffer slots than the runtime number of in‑flight loads. Smem is spent, latency is *not* hidden — you are paying the cost without the benefit. The extra buffers are filled but the scheduler cannot actually keep them all in flight because the wait granularity is too coarse. This is the failure mode that produces **‑0.9 %** vs the baseline: the kernel looks more pipelined (more stages, more smem) but is actually slower because the sync pattern doesn't match.

Concretely: if you have 3 buffer slots but your `cp.async.wait_group<2>` only keeps 2 loads in flight, the third buffer is decorative — its smem just squeezes the epilogue and increases register pressure. The fix is **not** to add stages; it is to align `wait_group` granularity (or `mbarrier` phase count) with `NUM_STAGES`.

Rule of thumb:

```
sync_count == NUM_STAGES - 1   # one slot being computed on, rest in flight
```

---

#### 7. When increasing NUM_STAGES helps — and when it doesn't

**Helps when:**

- `NUM_STAGES * T_compute < T_load` (load‑latency bound regime — see inequality §2).
- DRAM or L2 utilization is high (long‑latency loads, contention) so `T_load` grows beyond the static 200 cyc.
- There is **free smem** after the epilogue — i.e. increasing S does not force a tile‑size shrink.

**Doesn't help (and can hurt) when:**

- Already compute‑bound (TC ≥ ~90 %). Extra stages buy no throughput, see diminishing‑returns threshold `S*+1`.
- Smem is the constraint: pushing S past the budget forces `BLK_M`/`BLK_N` down, which **lowers arithmetic intensity** `2*M*N*K / (M*K + N*K + M*N)` and reduces TC occupancy — a net loss.
- **Sync count mismatch** (§6): more buffers than the wait pattern can actually overlap.
- The kernel is **barrier‑stall‑dominant with low TC** — adding stages without fixing the sync granularity amplifies the stall, it doesn't cure it.

---

#### 8. Concrete case: `v4-blk96` with S = 2 → **−0.9 %**

Reducing `BLK_M`/`BLK_N` to 96 and dropping to `NUM_STAGES = 2`:

```
S=2: 2 * 100 = 200  ≟  T_load = 200   ← exactly at the boundary
```

No slack. Any TMA jitter (L2 miss, DRAM row conflict, TMA‑unit contention from a neighbor SM) pushes a load past 200 cyc → WGMMA stalls at the barrier → TC drops below 90 %, **−0.9 %** vs the S=3 baseline. The smaller tile also lowered arithmetic intensity, so even at 100 % TC the per‑SM throughput was lower. Two compounding losses from one knob.

Lesson: **S = 2 is the knife‑edge; S = 3 buys the 100‑cyc slack that absorbs the real‑world variance.** Don't tune `NUM_STAGES` without simultaneously checking TC %, barrier‑stall share, and the smem headroom for the tile size you actually want.

---

#### 9. Quick tuning checklist

1. Compute `S* = ⌈T_load / T_compute⌉` (H20 FP16 TMA/WGMMA → `S* = 2`).
2. Pick `NUM_STAGES = S* + 1` for slack (→ 3) — **if smem fits**.
3. Verify `smem_total = S * (BLK_M*BLK_K + BLK_N*BLK_K) * bytes + smem_epilogue ≤ 228 KiB`.
4. Ensure **sync count == S − 1** (wait granularity matches buffer count, §6).
5. Run ncu: want TC ≥ 90 %, DRAM low, barrier stall dominant *with* high TC. If TC ≤ 50 % and barrier stall high → depth/sync mismatch, **don't** just bump S.
6. If TC is already ≥ 95 %, **stop**: more stages cannot help, only the tile size / MMA atom / epilogue can.

---

### 2. Tile Size vs Pipeline Depth

The fundamental budget equation. Shared memory is the hard constraint that links tile size to pipeline depth:

```
smem_per_block ≈ (BLK_M·BLK_K + BLK_N·BLK_K) · sizeof(elem) · NUM_STAGES  +  epilogue
              ≤  smem_per_SM                                  (H20: 228 KB)
```

For FP16 on our kernel (BLK_M=128, BLK_N=256, BLK_K=64, S=3):
`(128+256)·64·2·3 = 147 KB` of pipeline buffers, plus epilogue/double‑buffer → **214 KB total**, which fits exactly **one CTA per SM** (228 KB limit). This single fact drives almost every other number: occupancy collapses to **13.9%** (384 threads / 2048 max, capped to 1 CTA by smem), and with only one resident CTA the scheduler has nothing else to run when the resident warps block → **No‑Eligible 96.7%**.

The latency‑hiding condition. A pipeline hides producer (TMA) latency only if the consumer cannot outrun the buffered stages:

```
NUM_STAGES · max(TMA_cyc, ~0)  ≳  WGMMA_cyc_per_iter
```

But the *real* trade is arithmetic intensity, not just stage count. TMA fetches `(BLK_M·BLK_K + BLK_N·BLK_K)·2` bytes and WGMMA does `2·BLK_M·BLK_N·BLK_K` FLOPs on them, so the **bytes‑per‑FLOP ratio shrinks as the tile grows**:

| Tile (M·N·K) | Bytes/stage | FLOPs/stage | AI (FLOP/B) | smem/S3 |
|---|---|---|---|---|
| 128·256·64 (large) | 49 KB | 4.19M | 86 | 147 KB |
| 64·64·64 (small)   | 16 KB | 0.52M | 33 | 80 KB  (→ fits 2 CTA/SM) |

Large tile → high AI → WGMMA stays busy → **TC 90.9%**, but only S=3 fits, so when a stage isn't ready the consumer stalls hard → **CTA‑barrier stall 74.3%**. Small tile (v7, 64×64, S=5): half the smem → 5 stages → the next stage is almost always already arrived → **barrier stall 74%→8% (−82%)**, but each WGMMA does 8× less work, so launch/epilogue overhead and the relatively‑slower TMA dominate → **TC 91%→83% (−8%)**.

Measured net effect on the H20 vs cuBLAS:

| Problem | cuBLAS | large tile (S3) | small tile (S5) | Δ small vs large |
|---|---|---|---|---|
| 512³    | 21.1 T | — | — | **+13%** |
| 4096³   | 132.2 T | large wins | — | **−12%** |
| 16384³  | 139.2 T (≈94% of 148 T peak) | large wins | — | large |

**When it helps.** Small tiles win in the **latency‑bound / small‑N regime** (512³): pipeline stalls dominate and extra stages + 2× occupancy compensate for the TC efficiency loss. Large tiles win in the **compute‑bound / large‑N regime** (4096³+): TC utilization is the bottleneck and must be maximized. cuBLAS does exactly this — it heuristic‑dispatches tile shape by problem size, which is why a single fixed‑tile kernel always loses to it on at least one scale.

---

### 3. Epilogue Overlap

After the last WGMMA of a tile, the accumulator (in registers) must leave the SM: `rmoreg → smem` (R2S, warp‑level store) then `smem → gmem` (S2G, TMA store). On our kernel this epilogue costs **~180 cycles**, and during every one of them the Tensor Core is **idle** — no WGMMA can issue because the next tile's data isn't ready and the accumulator is still being drained.

The overlap trick. Issue the store but **do not wait for it** — `commit_group` on the TMA store *without* a matching `wait_group`. Immediately start the next tile's mainloop (next TMA prefetch + WGMMA). Only at the *next* epilogue do you call `wait_group(1)`, which waits for the oldest outstanding store to complete before reusing its smem store‑buffer slot.

With `NUM_STAGES` buffers this gives you exactly **1 iteration of slack** to hide the store. The overlap is net‑positive only when:

```
mainloop_cyc_per_iter  ≥  store_cyc                (≈180 cyc on H20)
```

If the inequality holds, the store completes during the next mainloop and is fully hidden. If it fails, the deferred `wait_group(1)` blocks immediately (the previous store is still in flight), you pay the store latency anyway, *and* you've created smem‑port contention with the next mainloop → net loss.

Concrete numbers from the kernel:

| Problem | mainloop/iter | store | overlap? | Δ vs sync‑epilogue |
|---|---|---|---|---|
| 4096³   | ~6400 cyc | 180 cyc | yes (6400 ≫ 180) | **+1.4%** |
| 512³, sk=8 | ~50 cyc | 180 cyc | no (50 ≪ 180) | **−4.2%** |

Theoretical ceiling: `min(store, mainloop) / (mainloop + store)`. For 4096³ that's `180/6580 ≈ 2.7%`; the measured +1.4% is roughly half because the overlap is imperfect (smem bank conflicts on R2S, TMA descriptor issue cost, and the final‑tile drain still serializes). For 512³ sk=8 the ceiling is `50/230 ≈ 22%` — but the inequality is violated, so instead of saving ~22% you lose 4.2%.

**When it helps.** Only in the **compute‑bound regime where mainloop ≫ store** (large tiles, deep K‑iteration count). At small N_K or with tiny tiles the store outlasts the mainloop and the trick backfires — just `wait_group` synchronously. The gain is also intrinsically capped near ~2–3% because the epilogue is a small fraction of total time, so at large scale expect only marginal improvement on top of an already‑90%+ TC kernel.

---

### 4. Per‑Warpgroup Barriers

The sync structure. Our kernel runs **3 warpgroups** (384 threads): WG0 = DMA producer (TMA), WG1+WG2 = MMA consumers (WGMMA). Producer and consumers synchronize per pipeline stage through an mbarrier: the producer does `arrive_expect_tx(bytes)` when it has issued the TMA for stage `i`, and each consumer `wait()`s on that barrier before reading stage `i` from smem.

The problem with one shared mbarrier. The DSL's `PipelineTmaAsync` uses a **single mbarrier** for `consumer_release` across *both* MMA warpgroups. Both WGs wait on the same barrier object with the same arrival count, so per stage:

```
WG1 wait ─┐
          ├─ single mbarrier ─→ released once, both wake together
WG2 wait ─┘
```

If WG1 finishes its WGMMA for stage `i+1` early and reaches the `wait()` for stage `i+2`, it cannot proceed until the producer has arrived *and* WG2 has also moved on — i.e., **WG1 is serialized to the speed of the slower of {producer, WG2}**. On a balanced 2‑WG consumer this is a small but real per‑stage skew (~a few % of one WGMMA duration), multiplied across all K‑iterations.

Per‑WG barriers fix this. Give each MMA warpgroup its own mbarrier; the producer arrives on **both** independently:

```
producer ──arrive──► mbar_A ──► WG1 wait   (independent)
producer ──arrive──► mbar_B ──► WG2 wait   (independent)
```

Now WG1 releases as soon as *its* data is ready, regardless of WG2's progress. Expected gain **~1–2%** — small because the two WGs are doing identical work and the skew is only the variance, not the mean.

Why we didn't ship it. The DSL abstraction `PipelineTmaAsync` hard‑codes one mbarrier over the shared smem pipeline and does not expose per‑WG mbarrier partitioning from inside `@cute.kernel` — the arrive/wait counts are inferred from the copy descriptor, not user‑settable per consumer. HPC‑Ops achieves it in raw **C++** by manually instantiating two `mbarrier` objects in smem and calling `mbarrier.arrive_expect_tx` / `mbarrier.wait` per WG with explicit transaction‑byte counts.

**When it helps.** Whenever you have ≥2 consumer warpgroups sharing one pipeline (always, for a 2‑WG‑MMA Hopper GEMM). Gain is bounded by inter‑WG skew, so ~1–2% in practice; unreachable from the DSL without dropping to C++.

---

### 5. Manual Barrier (vs PipelineTmaAsync)

What `PipelineTmaAsync` does. It is the high‑level DSL primitive for the producer side: you call `cute.copy(tiled_copy, gA, sA)` with a `TiledCopy` that has TMA semantics, and the pipeline *automatically* (a) issues the TMA `cp.async.bulk.tensor`, (b) `arrive_expect_tx` on the stage's mbarrier with the byte count it computed from the copy shape, and (c) lets the consumer `wait()`. One barrier, one arrive, one byte count, bundled for the whole `A+B` fetch of a stage.

What you lose. Because A and B are different tensors with different shapes (A is M×K, B is N×K), different TMA descriptors, and different byte counts, bundling them into one arrive/wait means:

- **The consumer can't start until *both* A and B have fully arrived**, even if one operand was ready first. For a tall‑thin A and fat B this serializes the ready‑earlier operand behind the ready‑later one.
- The arrive count is fixed (one producer arrive per stage), so you can't do **per‑WG** arrival (see §4) or partial arrival.
- You can't overlap A‑arrive with B‑issue — they're committed together.

Manual mbarrier. Drop to the low‑level ops and drive the barrier yourself:

```
mbarrier.init(sA_ptr, 1)
mbarrier.arrive_expect_tx(sA_ptr, bytes_A)      # A's TMA issued, arrive now
mbarrier.arrive_expect_tx(sB_ptr, bytes_B)      # B's TMA issued, arrive now (separate or same bar)
...
mbarrier.wait(sA_ptr, phase)                    # consumer waits per‑operand
```

Benefits: (1) **separate A vs B arrival** → release the consumer for the operand that's ready first; (2) **per‑WG arrive counts** (ties into §4 — each WG's mbarrier can require N arrives, enabling per‑WG release); (3) **finer‑grained overlap** of A‑arrive with B‑issue. Expected gain **~1–2%**, of the same order as §4 because the dominant serialization being removed is the same kind (a few cycles per stage of forced bundling).

Why we didn't ship it. `@cute.kernel` does not expose `mbarrier_init` / `arrive_expect_tx` / `wait` intrinsics — they are deliberately hidden behind `PipelineTmaAsync` to keep the Python‑DSL surface safe. To get them you write the kernel in **C++ (HPC‑Ops `FullBarrier`)**, where you own the mbarrier objects in smem and call the `mbarrier` PTX intrinsics directly.

**When it helps.** Any time you want operand‑level or WG‑level release granularity — i.e., exactly the situations where §4 helps. Stand‑alone gain is ~1–2%; combined with per‑WG barriers (§4) the two are somewhat additive because they remove different serializations, but both require leaving the DSL.

---

### 6. Warp Specialization (Producer/Consumer Split)

Without WS. A single warpgroup does **both** TMA issue and WGMMA, serially, in the same warps. Within one warp you cannot issue a TMA `cp.async.bulk` and a WGMMA in the same cycle, so the ~200‑cycle TMA latency lands on the critical path of every iteration:

```
iter (no WS):  TMA_issue ──200 cyc──► arrive ──► WGMMA(100 cyc) ──► TMA_issue ...
                                     ▲ TMA latency fully serial, TC idle 200 cyc ▲
```

With WS. Split the 3 warpgroups by role:

| WG | threads | role | regs/thread | why |
|----|---------|------|-------------|-----|
| WG0 | 128 (4 warps) | producer: TMA only | 24–40 | only needs descriptors + pointers, no accumulator |
| WG1 | 128 (4 warps) | consumer: WGMMA | ~232 | holds the accumulator + A/B register fragments |
| WG2 | 128 (4 warps) | consumer: WGMMA | ~232 | second consumer for 2× WGMMA issue rate |

Producer and consumer run **concurrently**, synchronizing per stage via the mbarrier from §4/§5. TMA latency is now hidden behind the consumer's WGMMA stream rather than sitting on the critical path:

```
WG0:  TMA(i) TMA(i+1) TMA(i+2) ...         (producer stays ahead)
WG1:  ........wait(i) WGMMA(i) wait(i+1) WGMMA(i+1) ...
```

Costs.
- **128 threads do no MACs** (1/3 of the CTA). But TC throughput is fed by the 2 consumer WGs (256 threads), which is enough to saturate the WGMMA issue rate — so the "lost" 128 threads weren't going to add TC FLOPs anyway; they would just have competed for the same instruction slots.
- **Register split is asymmetric, not transferable**: regs are per‑thread, so WG0's spare regs don't move to WG1. But the split means WG0 threads don't need an accumulator, so the CTA's reg profile is balanced to ~154–168/thread avg, fitting in `65536 / 384 = 170.7` regs/thread — just barely. Without WS, every thread would need accumulator regs → would blow the per‑SM reg limit and cut occupancy further.
- **Epilogue**: only the MMA WGs own the accumulator, so the R2S+S2G epilogue (§3) must run on WG1/WG2. Coordination uses a `NamedBarrier(256)` across just the two MMA WGs (not all 384), plus the producer↔consumer mbarrier for the mainloop.

Measured result. v1 (with WS): **TC 90.9%**, 130.6 T at large scale. A cluster‑style kernel that does TMA+WGMMA in the same warpgroup (no WS): 129.7 T. WS gain **+0.7%**, i.e. ~1% at large scale. Small because at 16384³ the deep pipeline (S=3 + large tile) already hides most TMA latency, so the WS‑induced overlap is worth little extra. Where WS pays more: **shallow pipelines / small tiles / latency‑bound regimes** (exactly where §2's small‑tile config lives) — there the producer/consumer split is the *only* thing hiding TMA latency, since NUM_STAGES can't.

**When it helps.** It is most valuable for **correctness and small‑tile / shallow‑pipeline regimes**: it gives a clean producer/consumer programming model and is the only latency‑hiding mechanism when stage depth is insufficient. At large scale with a deep pipeline it's ~1% and arguably more important as a structuring / correctness device than as a perf knob.

---

### 7. Split-K

**Idea.** When the natural tile grid `ceil(M/BLK_M) × ceil(N/BLK_N)` under-fills the GPU, slice the K-loop and distribute the slices across CTAs. The grid becomes `M_tiles × N_tiles × sk`; each CTA owns a K-slice of length `K/sk`, writes a partial accumulation `Cₚ` into a **private buffer** (`sk` partials per output tile), and the host does `torch.stack(...).sum(dim=0)` to combine them.

**Formula.**
```
tiles_base      = ceil(M/BLK_M) * ceil(N/BLK_N)
tiles_split     = tiles_base * sk
K_tiles_per_CTA = K / (BLK_K * sk)        # must be ≥ ~NUM_STAGES to amortize the pipeline
partial_buffers = sk  (per output tile)
```

**Why it works on small GEMMs.** Baseline `512³` with `BLK_M=128, BLK_N=256` yields only `4 × 2 = 8` tiles for 78 SMs → 70 SMs idle for the whole kernel. `sk=8` lifts that to `64` CTAs ≈ 0.82 CTAs/SM, recovering the idle silicon.

| M=N=K | baseline | best sk | split-K TFLOPS | vs baseline | vs cuBLAS |
|------|----------|---------|----------------|-------------|-----------|
| 512  | 12.1 T   | 8       | 42.9 T         | +254%       | +129% (cuBLAS 21.1 T) |
| 1024 | —        | 2       | 91.3 T         | —           | — |
| 2048 | —        | 4       | —              | —           | — |
| 4096 | —        | 4       | —              | —           | — |

**Non-monotonic scaling — the gotcha.** More `sk` is *not* always better. At `1024³`:
- `sk=2` → `K_tiles_per_CTA = 1024/(64×2) = 8` → 91.3 T ✅
- `sk=4` → `K_tiles_per_CTA = 1024/(64×4) = 4` → 84.4 T ❌ (slower)

With `NUM_STAGES=3`, each CTA spends the first `~3` and last `~3` K-tiles in pipeline fill/drain (no steady-state overlap). When `K_tiles_per_CTA` drops toward `NUM_STAGES`, the fill/drain fraction `2·NUM_STAGES / K_tiles_per_CTA` explodes and erodes the gain. Pick `sk` to keep `K_tiles_per_CTA ≫ NUM_STAGES` *and* to lift tile count above `~num_SMs`.

**Best `sk` by size:** `512→8`, `1024→2`, `2048→4`, `4096→4`. Always sweep; never assume monotonicity.

---

### 8. Persistent Kernel

**Idea.** Stop launching one CTA per tile. Launch `min(tiles, num_SMs)` CTAs and let each CTA loop over `ceil(tiles/num_SMs)` tiles, striding through the grid. This collapses the launch tail — the last partial wave where some SMs sit idle — and turns the kernel into a software-scheduled wave loop.

**Formula.**
```
launched_CTAs = min(tiles, num_SMs)              # 78 on H20
tiles_per_CTA  = ceil(tiles / num_SMs)
tail_waste     = (num_SMs - (tiles mod num_SMs)) / num_SMs   # when tiles mod num_SMs ≠ 0
```

**Worked example — `4096³`.** Tiles `= (4096/128) × (4096/256) = 32 × 16 = 512`. Waves `= 512/78 = 6.56`. Last wave carries `512 − 6·78 = 44` CTAs, so `34` SMs idle → tail-wave efficiency `44/78 = 56%`, whole-kernel SM-efficiency `512/(7·78) = 93.8%`. Persistent variant: launch `78` CTAs, each does `ceil(512/78) = 7` tiles, no tail. Theoretical ceiling: `+6.2%`.

**Actual result: `+1.3%` only.** Why the let-down? In CuTe DSL the per-tile loop body is guarded by a `staged-if` on the tile boundary; that conditional forces the **consumer (wgmma) pipeline to be re-set-up for every tile**. The re-setup overhead lands on the critical path of a kernel already at `TC = 90.9%` (compute-bound), eating most of the tail-wave saving.

| Size | Waves | Persistent gain |
|------|-------|----------------|
| 4096³ | 6.56 | +1.3% (marginal) |
| 512³ | 0.10 | 0% (can’t help — under one wave) |

**Takeaways.**
1. Persistent kernels *cannot* help when `tiles < num_SMs` (e.g. `512³` = 8 tiles → 0.1 waves).
2. The win shrinks toward zero as TC% → 100% (compute-bound); tail-wave math only matters when you have spare SM cycles.
3. The real value is **not** the tail elimination — it is that a kernel-owned tile loop is the *prerequisite* for **epilogue overlap**: overlapping the `C`-tile store with the next tile’s MMA compute, which needs the persistent structure to exist at all.

---

### 9. FP16 Accumulator + Occupancy

**Idea.** Store the MMA accumulator in FP16 instead of FP32. The accumulator fragment lives in registers for the whole K-loop, so halving its width cuts register pressure dramatically, opening the door to `2` blocks/SM.

**Register math (H20, 256 threads/block, 65536 regs/SM).**
```
FP32 acc: ~168 regs/thread  →  65536 / (1 × 256) = 256  → 1 block/SM  (168 ≤ 256)
FP16 acc: ~84  regs/thread  →  65536 / (2 × 256) = 128  → 2 blocks/SM (84  ≤ 128) ✓
```

**Result: occupancy doubled, TC unchanged (83% → 83%).** This breaks the "more blocks = more TC throughput" intuition, and it tells you something specific about Hopper:

- `wgmma` (warpgroup MMA) is a **long-latency async instruction issued to an SM-level tensor core** — not a per-warp FMA. A second resident block’s `wgmma` simply **queues behind** the first block’s `wgmma` on the same SM’s TC. No parallelism gain on the TC.
- The classic "many warps hide FMA latency" model does **not** apply: `wgmma` latency is hidden by the **TMA multi-stage pipeline** (`NUM_STAGES=3` producer/consumer overlap), i.e. *instruction-level* pipelining, not *warp-level* interleaving.
- For a compute-bound `wgmma` kernel, occupancy is nearly irrelevant — the TC is the bottleneck, and the TC is already saturated at one block/SM.

**Correctness cost.** FP16 accumulation truncates intermediate sums; the kernel reported `RE = 0.09–0.47%`, above the FP16-GEMM tolerance. The precision loss was unacceptable, so the change was **reverted** even though it "succeeded" at raising occupancy.

**Lesson.** Don’t chase occupancy on Hopper compute-bound kernels — measure TC%, and if it’s already ≥ ~85%, occupancy tuning is the wrong lever. The only honest path to 2 blocks/SM is reducing *real* consumer register usage (see §10/§11), and only if you have headroom on the TC.

---

### 10. `launch_bounds` / `min_blocks_per_mp`

**Idea.** `@cuda.annotate(max_blocks_per_mp=N)` (aka `launch_bounds` / `min_blocks_per_mp`) is a *compiler hint*: "I want N blocks resident per SM." The compiler honors it by **capping register allocation** so N blocks fit:

**Formula.**
```
max_regs_per_thread = floor( 65536 / (min_blocks_per_mp × threads_per_block) )
```

With `threads_per_block = 256`:
| `min_blocks_per_mp` | max regs/thread | kernel needs 168 | effect |
|---------------------|-----------------|-------------------|--------|
| 1 | 256 | 168 ≤ 256 | no-op (no constraint added) |
| 2 | 128 | 168 > 128 | ❌ force **40+ register spill** |

**The catastrophic cascade.** Forcing `min_blocks_per_mp=2` on a kernel that genuinely needs 168 regs doesn’t magic extra capacity into the register file — it makes the compiler **spill** the overflow to *local memory* (DRAM-backed). On a tightly-scheduled `wgmma` + TMA pipeline this is devastating:

```
spill → extra DRAM local loads/stores (≈400 cyc each)
      → wgmma operands not ready → async pipeline stalls
      → spill save/restore bloats code → I-cache misses
      → producer/consumer handshake can't progress → pipeline deadlock
```

**Measured impact: ~9000× slowdown** (`0.04 ms` → `360 s`). The kernel didn’t get "slightly slower" — it effectively hung.

**Lesson.** `launch_bounds` is a *declaration of intent*, not a *source of capacity*. To legitimately reach `2` blocks/SM you must **reduce register pressure at the source** — smaller fragments, fewer simultaneously-live accumulators, FP16 accumulator (if precision allows), reordering to shorten live ranges — *then* the compiler will fit `2` blocks naturally. Forcing the hint while the IR still needs 168 regs just relocates the shortfall into the worst possible place: DRAM spill inside a latency-sensitive async pipeline.

---

### 11. `reg_dealloc` / `reg_alloc`

**Idea.** Hopper `wgmma` programs partition the warpgroup’s register file between the **producer** (TMA-copy warps, issue `cp.async`/TMA loads) and the **consumer** (wgmma warps, issue `wgmma`). Two knob-style directives set the split:
- `reg_dealloc = P` — producer *releases* P registers per producer thread after its work.
- `reg_alloc   = C` — consumer *claims* C registers per consumer thread from the freed pool.

**Budget formula.** With `prod_threads` producer threads and `cons_threads` consumer threads sharing one SM’s 65536 regs:
```
consumer_max = (65536 − prod × prod_threads) / cons_threads
```
Worked: `prod=24, prod_threads=128, cons_threads=256` → `(65536 − 24·128)/256 = (65536 − 3072)/256 = 244`. Consumer asks `C=232 ≤ 244` ✓ fits.

**But the knob is a no-op here.** The catch: the consumer’s *actual* register usage is only `154–168`; `232` is a **ceiling/declaration**, not measured need. Because `244 (available) > 232 (declared) > 168 (actual)`, the consumer already has comfortable headroom. Lowering the producer `reg_dealloc` from `40` to `24` frees more regs on the producer side, but **those freed regs do not transfer** to the consumer’s budget — the two pools are independently bounded by the directive, and the consumer was never the constraint.

| Knob change | Effect |
|-------------|--------|
| `reg_dealloc 40 → 24` (producer) | **no-op** — consumer already had enough; producer savings don’t flow across |
| `reg_alloc 232 → 200` (consumer) | would actually free budget for a 2nd block, **iff** real usage ≤ 200 |

**Lesson.** `reg_dealloc`/`reg_alloc` are *declarative ceilings*, not a reallocation pump. Producer savings **do not** become consumer capacity automatically; the only way to relax the consumer’s footprint (and so enable 2 blocks/SM) is to cut the consumer’s *actual* `wgmma` register demand — same conclusion as §10. The knobs describe the budget; they don’t create headroom. Tune the consumer’s `reg_alloc` to match its true usage, and reduce that true usage at the source (fragments, accumulators, live ranges) if you need another resident block.

---

### 12. TMA Multicast + Cluster

**Mechanism.** A *cluster* groups N CTAs (cooperative thread arrays) that share a
Distributed Shared Memory (DSMEM) bus across the SMs they land on. A **multicast
TMA** load reads its source bytes from the memory hierarchy **once** and fans the
payload out into several CTAs' shared memory in one transaction. (It is *not* one
"leader" CTA fetching the whole tile for everyone: each member issues only its
`1/N` slice of the tile — see the wire-level picture below — and each slice is
broadcast to all members named in its `mcast_mask`.)

**Bandwidth savings formula.**

```text
gmem_reads_saved(N)  = N - 1            (per multicast tile)
DRAM_bytes(cluster)  = DRAM_bytes(nomcast) / N
```

For the H20 GEMM baseline (BLK_M=128, BLK_N=256, BLK_K=64, NUM_STAGES=3, 384 threads), DRAM utilization is already only **7.2%** with **TC 90.9%** and **L2 hit 66%** — the kernel is compute-bound, so the saved gmem bandwidth has no place to convert back into speed.

**Critical correction of a common misconception.** "Multiple CTAs doing one MMA cooperatively" is **wrong**. `wgmma` (warpgroup MMA) is a strictly **single-CTA** instruction — it operates on one CTA's rmem and smem. The cluster primitive enables **data sharing** (one read, many smem copies), *not* **compute cooperation**. Confusing the two leads to invalid schedules where different CTAs try to feed a single MMA.

**Expected speedup by regime.**

| Regime | DRAM util | Gain |
|---|---|---|
| Compute-bound | ≤ 5% | ~0–2% |
| Memory-bound | ≥ 40% | +10–20% |

**Implementation status.** RESOLVED in CuTe DSL 4.7.0 — shipped as
`gemm-v9-multicast` (#18, one-way) and `gemm-v10-bimcast` (#19, bidirectional +
2×2). The earlier "nine approaches, all failed" verdict (#17) was self-inflicted:
the real blockers were a wrong `slice_` axis on the shared mode (double-issue →
RE≈70%) and unpinned layout strides (rank/coord swap → only bites at true 2×2).
The lesson-12 recipe below is the complete working form.

**Which operand is multicast — one picture, then three rules.**

The whole topic reduces to one identity: a CTA at grid slot (m, n) computes
`C[m,n] = A[m,:] · B[:,n]` — **its A tile depends only on the row m, its B tile
only on the column n.** A cluster is just a small rectangle on the (m, n) tile
grid; whatever is *identical inside that rectangle* is what multicast dedups.

A 2×2 cluster on the tile grid (each cell = one CTA, listing what it needs):

```text
              column n=0          column n=1
           ┌────────────────┬────────────────┐
 row m=0   │  CTA(0,0)      │  CTA(0,1)      │
 (needs A0)│  needs A0, B0  │  needs A0, B1  │◄─── same ROW  ⇒ same A0
           ├────────────────┼────────────────┤
 row m=1   │  CTA(1,0)      │  CTA(1,1)      │
 (needs A1)│  needs A1, B0  │  needs A1, B1  │
           └────────────────┴────────────────┘
                 ▲                ▲
                 └──── same COLUMN ⇒ same B0 ┘
```

Rule 1 — **stack members along M** (`CLUSTER_M = 2`, grid.y): members share a
column ⇒ share **B** ⇒ multicast **B** (this is what `gemm-v9` shipped: the
`(2,1)` config pairs CTA(0,0)+(1,0), both need B0).
Rule 2 — **line members along N** (`CLUSTER_N = 2`, grid.x): members share a row
⇒ share **A** ⇒ multicast **A**.
Rule 3 — **2×2**: both directions share something ⇒ multicast **both** (A along
rows, B along columns; `gemm-v10` verified).

What multicast actually does at the wire level, for one shared tile (say B0,
split into halves b0ᵀ/b0ᴮ for the 2 members, `num_multicast=2` + mask {0,1}):

```text
without multicast:  CTA0 reads B0 completely ──► smem0        (B0 fetched twice)
                    CTA1 reads B0 completely ──► smem1        (DRAM/L2 answers 2x)

with multicast:     CTA0 issues b0ᵀ ──► TMA reads it ONCE ──► lands in smem0 AND smem1
                    CTA1 issues b0ᴮ ──► TMA reads it ONCE ──► lands in smem0 AND smem1
                    each CTA still ends with the FULL B0 in its own smem;
                    L2/DRAM answered exactly one B0 worth of bytes, total.
```

So the three ingredients must tell one consistent story: the **atom** knows the
fan-out width (`num_multicast`), `tma_partition(cta_coord, cta_layout)` makes
each member issue only its 1/N slice, and `mcast_mask` names the receiving
members. Get any one of them inconsistent with the rectangle in picture 1 and
you get double-issue (RE≈70%) or rank swap (RE≈122%) — caveats below.

#### Multicast+cluster kernel vs ordinary kernel — the full delta

(`gemm_kernel_cluster.py`; "ordinary" = same code with `CLUSTER_M=CLUSTER_N=1`.)

| Component | Ordinary | Multicast + cluster | Why |
|---|---|---|---|
| launch | `grid=(grid_n, grid_m, 1)` | + `cluster=(CLUSTER_N, CLUSTER_M, 1)` | cluster dims must mirror grid axes (x fastest); members co-schedule on DSMEM-connected SMs |
| CTA identity | `block_idx()` | + `block_idx_in_cluster()` (linear rank, x-fastest) | feeds every cluster-aware lookup |
| rank→coord | — | `cta_layout_mnk.get_flat_coord(rank)`, strides **pinned n-fastest** | CuTe default is m-fastest → swaps m/n at 2×2 (see caveat 1) |
| load atom | `G2SOp` | `G2SMulticastOp` + `num_multicast=<axis>` on the shared operand | plain flavor hard-rejects `num_multicast≠1` (helpers.py:523); SASS shows `UTMALDG.2D.MULTICAST` |
| partitioning | `tma_partition(atom, 0, layout(1), …)` | shared axis: `tma_partition(atom, cta_crd, cta_layout, …)` | each member issues only its slice instead of the whole tile |
| copy call | `cute.copy(atom, …)` | + `mcast_mask=` (Int bitmask or `Int16(0)` on degenerate axis) | tells HW which members receive the slice |
| smem | private | private *capacity* unchanged; TMA may now write peer smem | multicast widens reach, not size |
| stage barrier arrives | `NUM_WARPS` | `(CLUSTER_M+CLUSTER_N-1) * NUM_WARPS` | barrier must see every contributing member's warps; miscount ⇒ hang or stale reads |
| `cta_layout_vmnk` (pipeline) | `(1,1,1,1)` | real cluster shape, pinned strides | pipeline routes `dst_rank`/mcast signaling through it |
| cluster start gate | none | `pipeline_init_arrive(cluster_shape_mn=…, relaxed=True)` + `wait` | no producer may issue before all members arrived |
| `tx_count` | stage bytes | **unchanged** (do not divide by cluster size) | barrier fires when the whole tile has landed |
| MMA / epilogue | `wgmma` + TMA store | identical | cluster shares DATA only — wgmma stays single-CTA |

#### Caveats — every one of these actually bit us (#17→#19)

1. **Stride order of `cta_layout_mnk` (2×2 killer).** `make_layout((M,N,1))` is
   column-major (m fastest) but hardware ranks are grid-x = **n fastest**
   (`rank = n + CLUSTER_N*m`). Degenerate axes hide the mismatch — `(2,1)` and
   `(1,2)` pass "by luck", `(2,2)` explodes with RE≈122%. Pin strides explicitly.
2. **Slice the correct axis.** B multicast ⇒ shared axis is M ⇒
   `slice_(cta_layout_mnk, (None,0,0))` + `cluster_coord_mnk[0]`; A multicast is
   the mirror `(0,None,0)`/coord[1]. Wrong branch ⇒ every member issues the full
   tile ⇒ duplicate writes, RE≈70% (bit #18 once, #19 again identically).
3. **Mask must match real sharing.** If grid/cluster/cta_layout ever get
   rearranged independently, the mask silently stops describing identical
   operands — data corruption with no error anywhere. Keep the three in one
   mental picture (diagram above).
4. **Arrive-count semantics.** `mcast_size = CLUSTER_M+CLUSTER_N-1` (union of
   contributors); the hand-rolled `mcast_size*NUM_WARPS` must equal the DSL's
   `enable_multicast_signaling` computation if you switch to it (verified E1, #19).
5. **Co-scheduling is a real cost.** Cluster members must be co-resident on one
   GPC. Free for compute-bound kernels; when the cluster is the wrong
   granularity (FA v12: merge colocated into the cluster) the gang-scheduling tax
   measured +0~35%. Cluster ≠ automatic win.
6. **Sweep the non-degenerate config.** `(2,1)`/`(1,2)` alone cannot catch
   rank-order bugs; `(2,2)` is the test that actually exercises the mapping.

#### Parameter dictionary

| Symbol | Value in this kernel | Meaning |
|---|---|---|
| `CLUSTER_M` | 2 | cluster extent along M-tiles (= grid.y). >1 ⇒ B shared ⇒ B multicast |
| `CLUSTER_N` | 1..2 | cluster extent along N-tiles (= grid.x). >1 ⇒ A shared ⇒ A multicast |
| `CLUSTER_SIZE` | `M*N` | CTAs per cluster |
| `cluster=(CLUSTER_N, CLUSTER_M, 1)` | launch | HW dims (x,y,z) mirroring grid; x=n fastest |
| `cta_layout_mnk` | `(CLUSTER_M, CLUSTER_N, 1) : (CLUSTER_N, 1, 1)` | rank↔(m,n,k) bijection; **strides are load-bearing** |
| `cta_layout_vmnk` | `(1, CLUSTER_M, CLUSTER_N, 1)` (pinned) | pipeline's cluster view; V=1 = no CTA-level TMA multicast *of the pipeline itself* |
| `block_idx_in_cluster()` | Int | linear rank ∈ [0, CLUSTER_SIZE) |
| `cluster_coord_mnk` | `get_flat_coord(rank)` | this CTA's (m,n,k) slot in the patch |
| `make_layout_image_mask(layout, coord, mode)` | Int16 bitmask | set of members sharing `coord` when mode axis is projected out; mode 0 ⇒ B's column set, mode 1 ⇒ A's row set |
| `a_mcast_mask` / `b_mcast_mask` | mask or `Int16(0)` | passed to `cute.copy(..., mcast_mask=)`; 0 = this operand isn't multicast |
| `num_multicast=` | axis size | TMA engine: how many copies of each issued slice to fan out |
| `G2SMulticastOp` | atom class | only class that lowers to `UTMALDG.*.MULTICAST` |
| `mcast_size` | `CLUSTER_M+CLUSTER_N-1` | contributors to one stage barrier (union, minus double-counted self) |
| `tx_count` | full A+B tile bytes | expected barrier bytes — never divide by cluster size |
| `pipeline_init_arrive/wait(cluster_shape_mn)` | relaxed pair | cluster-wide start gate before first producer issue |
| `defer_sync=True` | pipeline opt | skip built-in fence/sync; required for correct multicast staging (#17 limitation 2) |

---

### 13. L2 Cache Pinning

**Mechanism.** The CUDA driver exposes a stream-level cache policy via `cuStreamSetAttribute(stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &policy)`. The `CUaccessPolicyWindow` struct names a byte range `[base_ptr, base_ptr+num_bytes)` and a `hitRatio` ∈ [0,1]; the L2 then biases evictions to **persist** that region across the window's lifetime. For GEMM, the output matrix C is the natural pin target because every K-iteration writes back into the same M×N tile repeatedly.

**Bandwidth formula.**

```text
effective_L2_hit(C)  = min( hitRatio, L2_size_pinned / |C_tile| )
DRAM_writeback(C)    = |C| * (1 - effective_L2_hit(C))
```

H20 L2 = **60 MB**. A single 4096³ fp16 output tile is 4096×4096×2 B = **32 MB**, which fits in L2 — so pinning converts repeated write-backs into L2 hits.

**Measured results.**

| Problem size | Output tile | Δ throughput | Cause |
|---|---|---|---|
| 4096³ | 32 MB | **+0.9%** | write-back persistence improves |
| 512³  | 512 KB | **−0.6%** | L2 pollution evicts A/B, hurts reads |

**Verdict.** Marginal for compute-bound kernels (DRAM < 5%). The H20 baseline sits at DRAM 7.2% and TC 90.9%, so the headroom for any L2-side optimization is fundamentally capped near 1%. Pinning only pays when the **output reuse × output size** product is large enough to amortize the eviction cost on inputs.

---

### 14. L2 Compression

**Mechanism.** Hopper (sm_90) L2 has hardware **automatic line compression**: each cache line is compressed on fill and decompressed on access, transparent to software. Lines with low entropy (many repeated bytes, runs of zeros) compress to a fraction of their physical size, so more logical data fits in the same physical L2 capacity.

**Effective capacity formula.**

```text
L2_eff = L2_phys / mean(compression_ratio),   mean(compression_ratio) in (0, 1]
```

For random fp16 noise, `mean(compression_ratio) ~= 1` (incompressible). For sparse/zero-padded data it can drop to ~0.5 or lower.

**Empirical results on the GEMM workload.**

| Data pattern | Observed gain | Reason |
|---|---|---|
| Random fp16 (default benchmark) | **0%** | high entropy, no compression |
| ncu *estimated* ceiling | ~1.5% | if compression worked |

**Key constraints.**
- **No software API.** Compression is purely hardware; you cannot hint, control, or query it.
- **Conditional benefit.** Only helps *structured / low-entropy* data: sparse matrices, zero-padded NLP tensors, block-quantized (INT4/FP8) weights, masked attention regions. Dense random GEMM sees nothing.

For the H20 baseline (DRAM 7.2%, L2 hit 66%, L2 phys 60 MB), even a perfect 2× compression would only lift the effective L2 to 120 MB — and since the kernel is TC-bound (90.9%), the extra L2 capacity cannot translate into FLOPs.

---

### 15. K-loop Unroll

**Mechanism.** The K-reduction loop iterates `K // BLK_K` times. Unrolling by N emits N consecutive `cute.gemm(...)` invocations per source-level iteration. Because each `wgmma` is an *asynchronous* instruction, the instruction scheduler can issue all N before the first completes — instruction-level parallelism (ILP) fills the warp-scheduler bubbles that occur between dependent MMAs.

**ILP vs I-cache tradeoff formula.**

```text
speedup(N)       ~=  N / (N - bubbles_rel)   -  Icache_miss_penalty(N)
                     ~~~~~~ ILP gain ~~~~~~     ~~ code-size cost ~~
loop_body_size(N) ~=  N * |wgmma_seq|
```

Hopper has a **48 KB per-SM instruction cache**. A single WGMMA instruction sequence is ~100+ cycles and encodes to a sizable instruction footprint; doubling it (N=2) nearly doubles the loop body's I-cache demand.

**Measured results.**

| Unroll N | 512³ | 4096³ | Verdict |
|---|---|---|---|
| 1 (baseline) | — | — | best overall |
| 2 | **+1.0%** | **−0.9%** | small kernels win, large lose |
| 4 | worse | worse | I-cache thrash everywhere |

**Interpretation.**
- **512³:** the loop executes many times over a *small* working set; ILP gain (1.0%) exceeds the I-cache pressure.
- **4096³:** the same loop body now competes with the larger epi/mainloop code; I-cache misses cost more than ILP recovers (−0.9%).

Because each WGMMA ≈ 100+ cycles, the scheduler already has substantial latency-hiding depth at N=1; the marginal ILP from N=2 rarely beats the I-cache cost at production problem sizes. **Default: do not unroll.**

---

### 16. Double Accumulator (Ping-Pong)

**Mechanism.** Maintain two register-memory accumulators `C0`, `C1`. On even K-iterations, accumulate into `C0` while `C1` is being drained to shared memory (R2S) and stored to gmem via TMA (S2G); on odd iterations, swap roles. The goal is to overlap the long **TMA store tail** of one tile with the **MMA compute** of the next tile.

**Overlap window formula.**

```text
overlap_available = min(drain_cost, fill_cost)
drain_cost = R2S + TMA_S2G ~= 20 cyc (LSU)
fill_cost  = WGMMA         ~= 100+ cyc (TC)
=> overlap_available ~= 20 cyc
```

The drain path is **5× shorter** than the fill path, so the ping-pong only recovers ~20 cycles per tile — far less than the **~180-cycle TMA-store overlap** the existing epilogue/mainloop overlap already captures. Marginal upside.

**Why it is not implemented in the DSL.**
- The CuTe DSL **staged-if** construct performs *compile-time* selection between tensor variables; it cannot express a runtime `tile_iter % 2` index that alternates between two rmem tensors across loop iterations.
- A workaround would require **duplicating the entire mainloop + epilogue** into two static branches (one per accumulator), doubling compile time and code size, with the ~20-cyc ceiling above capping the upside.

**Verdict.** Conceptually valid for memory-bound kernels where drain ≈ fill, but for this Hopper WGMMA kernel (TC 90.9%, DRAM 7.2%) the arithmetic intensity already saturates the tensor cores, leaving the ping-pong trick with <1% headroom. Skip unless the store path becomes the bottleneck.

---

### 17. Block Swizzle (tile reordering)

**Goal.** Reorder the `(M-tile, N-tile)` CTA schedule so consecutive CTAs reuse each other's input stripes in L2, instead of thrashing L2 with disjoint working sets.

**Default M-major order (the baseline scheduler).** For each M-block `m`, stream all N-blocks `n = 0..N/BLK_N-1`. Per-stripe footprint at 4096³:

```
A-stripe = BLK_M × K × sizeof(FP16) = 128 × 4096 × 2 = 1 MiB
B-stripe = BLK_N × K × sizeof(FP16) = 256 × 4096 × 2 = 2 MiB
```

Resident working set (5 active M-blocks, 16 N-blocks, deduped B-stripes):

```
WS_M-major = 5 × 1 MiB + 16 × 2 MiB = 5 + 32 = 37 MiB   (< 60 MiB L2 ✓)
```

Adjacent M-tiles for the same `n` share the **same B-stripe** → B stays resident in L2; A is re-streamed per M-tile (cheap, 1 MiB). M-major therefore already maximizes A-row reuse by construction.

**Group-M swizzle (GROUP_M=4).** Schedule `GROUP_M` consecutive M-tiles before striding N:

```
(m=0,n=0),(m=1,n=0),(m=2,n=0),(m=3,n=0),
(m=0,n=1),(m=1,n=1), ...
```

Active footprint grows (4 current M + 2 in-flight overlap):

```
WS_GROUP_M=4 ≈ 6 × 1 MiB + N-stripes ≈ 6 + 32 = 38+ MiB   (more L2 pressure)
```

**Measured (H20, 4096³, SW128):**

| Scheduler        | L2 hit | TC    | DRAM  |
|------------------|--------|-------|-------|
| M-major (default)| 66 %   | 90.9 %| 7.2 % |
| GROUP_M=4        | 62 %   | 90.9 %| 7.2 % |

L2 hit **drops 66 %→62 %**: more resident M-blocks = larger working set, and at 4096³ the inputs (A+B = 64 MiB) already saturate the 60 MiB L2, so any extra MiB evicts useful lines.

**When it helps.** Only memory-bound kernels where (a) DRAM is the bottleneck (`DRAM % ≫ TC %`) and (b) total working set fits comfortably in L2. For this H20 GEMM — TC-bound at 90.9 %, DRAM only 7.2 % — L2 is not the bottleneck → group-M is net negative; the default M-major scheduler wins.

---

### 18. Padding for Non-Aligned Shapes

**Problem.** TMA loads a fixed `(BLK_M, BLK_K)` box from gmem in one transactional descriptor read. When `K` is not a multiple of `BLK_K`, the last K-tile overhangs the true K extent:

```
K = 333, BLK_K = 64
num_K_tiles = ceil(333 / 64) = 6
last tile covers K[320 : 384]   (64 wide)
valid range   K[320 : 333]     (13 valid)
OOB garbage   K[333 : 384]     (51 garbage elements)
```

TMA does **not** zero-pad OOB (unlike `cp.async` with `.zfill`). The 51 OOB elements are whatever bytes live past the allocation and feed directly into the MMA as real data.

**Symptom.** `run_gemm.py` prints:

```
RE = 141 %   → Failed
```

51/333 ≈ 15 % of the K-axis is garbage, weighted by whatever those bytes happen to be → relative error comparable to the true GEMM magnitude.

**Why 3000³ "worked" before.** `3000 % 64 = 32`, so the last tile overhangs by 32 elements. By luck those 32 gmem bytes happened to be zeros (allocator alignment fill), so the MMA accumulated `+0` and correctness held. This is **luck, not correctness** — shift the allocation or change allocator behavior and it breaks (as 333 demonstrates).

**Fix.** Pad inputs and output to tile multiples, run the kernel, then slice back:

```python
M_pad = ceil(M, BLK_M) * BLK_M
K_pad = ceil(K, BLK_K) * BLK_K
a = F.pad(a, (0, K_pad - K))  # (M, K_pad)
b = F.pad(b, (0, K_pad - K))  # (N, K_pad)
c = torch.zeros(M_pad, N_pad, ...)
run_kernel(a, b, c, M_pad, N_pad, K_pad)
c = c[:M, :N]  # restore original shape
```

Padding guarantees every TMA box reads in-bounds real data (the padded region is explicit zeros, contributing `+0` to the accumulation).

**Cost.** Correctness fix, not a performance lever. Worst-case padding ≤ `(BLK − 1)/BLK ≈ 50 %` extra work on the smallest dimension, amortized to ~0 % at large shapes. Apply whenever `K % BLK_K ≠ 0` or `M % BLK_M ≠ 0` or `N % BLK_N ≠ 0`.

---

### 19. BLK_K Increase (64 → 96) + Fewer Stages (3 → 2)

**Lever.** Larger `BLK_K` reduces the K-loop trip count and thus barrier/sync count:

```
num_K_tiles(64) = ceil(4096 / 64) = 64
num_K_tiles(96) = ceil(4096 / 96) = 43
Δ syncs = (64 − 43) / 64 = −33 %
```

**Cost (smem pressure).** Each pipeline stage holds one A-box + one B-box:

```
stage_size(64) = BLK_M·BLK_K·2 + BLK_N·BLK_K·2
               = 128·64·2 + 256·64·2  = 16 KiB + 32 KiB = 48 KiB
stage_size(96) = 128·96·2 + 256·96·2 = 24 KiB + 48 KiB = 72 KiB   (+50 %)
```

H20 smem budget = 228 KiB. With 214 KiB already committed by the CTA:

```
NUM_STAGES(64, 48 KiB) = floor(214 / 48) = 4  →  configured = 3  (room)
NUM_STAGES(96, 72 KiB) = floor(214 / 72) = 2  →  configured = 2  (maxed)
```

Increasing `BLK_K 64→96` therefore **forces the pipeline depth 3→2**.

**Trade-off.** A 3-stage mainloop overlaps `i+3` gmem→smem TMA copy, `i+2` rmem load/wait, and `i+1` rmem→rmem MMA. With only 2 stages the producer can no longer fully hide consumer latency → one bubble per K-iteration. **Lost overlap > saved syncs.**

**Measured (4096³):**

```
v4-baseline (BLK_K=64, STAGES=3, SW128):  TC = 90.9 %   TFLOPS = ref
v4-blk96    (BLK_K=96, STAGES=2, SW64):   TC ≈ 90 %     TFLOPS = ref × 0.991   (−0.9 %)
```

**Secondary effect: swizzle granularity.** `BLK_K=64` selects SW128 (1024-bit unit); `BLK_K=96` selects SW64 (512-bit unit, finer granularity, marginally more swizzle compute). Both remain bank-conflict-free (see §20).

**Conclusion.** Pipeline depth `NUM_STAGES` is the **single biggest contributor** to the 90.9 % Tensor Core throughput — the 3-stage overlap hides the ~600-cycle WGMMA latency behind TMA+rmem copy. Reducing sync count by 33 % does not compensate for collapsing the pipeline to 2 stages. Keep `BLK_K=64, NUM_STAGES=3`.

---

### 20. Swizzle Patterns (Bank Conflict Avoidance)

**Bank geometry.** Hopper shared memory = **32 banks × 4 B/bank = 128 B per cycle**. A naïve row-major layout places a 128-B `ldmatrix`/rmem-load vector in 8 consecutive 4-B words hitting the same 8 banks:

```
row r:  bank b  bank b  ... bank b   (8 × 16 B = 128 B, all in 8 banks)
        → 8-way bank conflict → 8 cycles instead of 1
```

**Swizzle.** XOR the row index into the column offset:

```
swizzled_col(r, c) = c  XOR  f(r)
```

Adjacent rows land in different banks → 8 words spread across all 32 banks → 0 conflicts, 1 cycle/transfer.

**Granularities** (selected by `get_smem_layout_atom`):

| Pattern | Unit (bits) | Unit (bytes) | Layout        |
|---------|-------------|--------------|---------------|
| SW128   | 1024        | 128 B        | 8 rows × 16 B |
| SW64    | 512         | 64 B         | 4 rows × 16 B |
| SW32    | 256         | 32 B         | 2 rows × 16 B |

Selection rule (verified in the installed DSL):

```
atom = get_smem_layout_atom(ROW_MAJOR, dtype=Float16, major_mode_size=BLK_K)
# BLK_K=64 → 64 × 16b = 1024 bit → SW128
# BLK_K=96 → 96 × 16b = 1536 bit → SW64 (round down to power-of-2 unit)
```

**Measured.** `ncu --section MemoryWorkloadAnalysis_Shared`:

```
Bank conflicts per shared-mem transaction: 0.00   (SW128, baseline)
```

**Why conflicts don't appear elsewhere.**

- **WGMMA reads smem via the descriptor path** — bypasses the LSU/bank-arbitration path entirely; descriptor issues async gmem-style accesses, no per-thread bank arbitration.
- **TMA writes smem via the DMA path** — bulk async copy, no bank arbitration.
- **Only the epilogue `R2S` (rmem→smem) / `S2G` (smem→gmem) use the LSU path**, and there the layout is *derived* from the swizzled atom to stay conflict-free automatically.

**Takeaways.**
1. Swizzle is **required for correctness** — the WGMMA descriptor expects the exact bit-pattern produced by `get_smem_layout_atom`; an unswizzled or mismatched layout yields wrong MMA operands.
2. Swizzle is **free for performance** — eliminates what would otherwise be 8-way bank conflicts on every `ldmatrix`, at zero extra instruction cost (the XOR is folded into address generation).
3. The pattern (SW128/SW64/SW32) is **derived from dtype and `BLK_K`**, never chosen by hand.

---

### 21. K-Major Data Layout

**GEMM convention.** `C = A · Bᵀ` with `A ∈ R^{M×K}`, `B ∈ R^{N×K}` (B stored row-major as N×K so K is the fast axis), `C ∈ R^{M×N}`.

**K-major.** The K dimension is **contiguous in memory** — successive K elements occupy adjacent addresses. `A(M, K)` row-major: `&A[m, k+1] − &A[m, k] = 1 element` → K is fast → K-major. Likewise `B(N, K)` row-major is K-major.

**Why the MMA needs K-major.** The WGMMA accumulates over K:

```
C[m, n] = Σ_{k} A[m, k] · B[n, k]
```

and strides along K on both operands. `cute.nvgpu.warpgroup.MmaF16BF16Op(..., OperandMajorMode.K)` declares the *major* (fast-varying) axis of each operand is K. The hardware streams K-contiguous vectors and fuses multiply-adds without address arithmetic on the K stride.

**Why TMA needs K-major.** TMA loads a contiguous `(BLK_M, BLK_K)` box. If K is contiguous in memory (K-major), one TMA descriptor issues **coalesced** transfers — `BLK_K` consecutive FP16s in adjacent addresses pack into minimal 128-B sectors. If layout were M-major (M fast), the same box would be `BLK_M` disjoint K-strided rows → uncoalesced, `BLK_M×` more sectors, `BLK_M×` the DRAM transactions.

**Concrete (baseline kernel).**

```python
mA = cute.make_tensor(ptr_a, cute.make_layout((M, K), stride=(K, 1)))  # row-major → K-fast
mB = cute.make_tensor(ptr_b, cute.make_layout((N, K), stride=(K, 1)))  # row-major → K-fast
```

Both match `OperandMajorMode.K`. The `make_cute_tensor(t)` helper in `common/cute_runtime.py` forwards the torch tensor's existing strides, so passing contiguous `(M, K)` and `(N, K)` torch tensors is sufficient.

**Correctness invariant.**

```
layout(A).fast_axis == K   ⟺   WGMMA op's MajorMode.K   ⟺   TMA coalesced
```

Break any link and the kernel produces garbage:
- Pass `A` as `(K, M)` (M-major) but tell the op `MajorMode.K` → MMA reads wrong operand → `RE ≫ tol`.
- Mismatch `B`'s major axis → same.
- Transpose one operand only → silent wrong-accumulator.

**M-major is a trap.** Learners sometimes store `A` as `(K, M)` "because the K loop is innermost and we want K contiguous in the *other* direction." Wrong: K-major means K contiguous **in memory**, not "K is the inner loop variable." The inner loop is always K (that's what the MMA does); the **memory layout** must put K adjacent so those inner-loop loads are coalesced.

---

### 22. CUDA Graphs

**Launch overhead anatomy.** Every `cudaLaunchKernel` traverses the driver API: argument marshalling, stream submission, context validation, and kernel-descriptor setup. Measured cost on H20: **5–10 µs per launch**, serialized on the launch thread.

**Overhead fraction** = `t_overhead / (t_overhead + t_kernel)`:

| Problem | Kernel runtime | Overhead fraction (5–10 µs) |
|---------|---------------|-----------------------------|
| 512³    | ~20 µs (0.02 ms) | **25–50 %** |
| 1024³   | ~150 µs        | 3–6 % |
| 4096³   | ~1.5 ms        | 0.3–0.7 % |
| 16384³  | ~24 ms         | <0.05 % |

Rule of thumb: launch overhead is only material for **sub-100 µs** kernels. Above that it disappears into the noise — which is why our 4096³/16384³ numbers are clean but the 512³ point looks artificially slow.

**CUDA Graphs.** Capture the launch sequence once (`cudaStreamBeginCapture` → run launches → `cudaStreamEndCapture` → `cudaGraphExec`), then replay with a single `cudaGraphLaunch` ≈ **1 µs**. All kernel arguments (pointers, scalars) are **baked in at capture time**; the graph replays the exact same addresses and parameter block.

**Incompatibility with our CuTe DSL path.** The compiled DSL function is dispatched through TVM-FFI: each call wraps `torch.Tensor` into `cute.Tensor` via `from_dlpack`, which **constructs a fresh `cute.Tensor` object per invocation**. Even though the underlying `torch.Tensor` data pointer is stable, the FFI conversion re-extracts pointers at call time, and graph capture requires them fixed at capture time. Result: `--cuda-graphs` cannot bind to the `cute.compile` callable.

**Auto-fallback.** When `--cuda-graphs` is requested but the compiled kernel is incompatible, we fall back to `cuda_bench` (manual `cudaEvent` timing around each launch). Reported numbers are then *kernel time + launch overhead* — for large problems indistinguishable from pure kernel time, but for 512³ it inflates the measurement by 25–50 %, exactly the regime where graphs would have helped.

**Practical takeaway.** For the 512³ PERFLOG point, the apparent inefficiency vs cuBLAS (21.1 T vs theoretical peak) is partly launch overhead, not TC utilization. CUDA Graphs would close that gap. For 4096³+ graphs are a no-op.

---

### 23. Fused Epilogue (GEMM + Activation)

**The unfused cost.** Many workloads apply a pointwise op after the GEMM:
- `C = ReLU(A·B + bias)`
- `C = GELU(A·B)`
- `C = softmax(A·B)` (attention)

Naive two-kernel pipeline:

```
GEMM kernel:  reg(C) ──R2S──▶ smem(C) ──TMA S2G──▶ gmem(C)            [write 1]
ACT  kernel:  gmem(C) ──TMA G2S──▶ smem ──S2R──▶ reg(C)              [read]
              reg(C) ──act()──▶ reg(C) ──R2S──▶ smem ──S2G──▶ gmem(C)  [write 2]
```

That is **one extra gmem round-trip + one extra kernel launch** for the intermediate `C`.

**Quantifying on 4096³ FP16.** `C` = `4096² × 2 B = 33.5 MB`. H20 HBM3 bandwidth ≈ 4 TB/s, so one round-trip (read + write):

```
t_rt = 2 × 33.5 MB / 4 TB/s ≈ 16.8 µs
t_launch ≈ 5–10 µs
t_epilogue_penalty ≈ 22–27 µs
```

Against a 1.5 ms GEMM that is ~1.5 % — small but real, and it scales linearly with `M·N`.

**Fused path.** Apply the activation **in registers** right after the WGMMA accumulators produce `D`, before any shared-memory or gmem write:

```python
# After the K-loop's final cute.gemm(tiled_mma, tCrD, tCrA, tCrB, tCrD)
acc = tCrD.load().to(out_dtype)  # WGMMA accumulator → out dtype
acc = activation(acc)  # pointwise, in registers
tCrD.store(acc)  # back into the store fragment
cute.copy(r2s_copy, tCrD, sC)  # R2S
cute.copy(s2g_tma, sC, mC)  # TMA S2G — single gmem write
```

Single gmem write, single kernel launch. Activation cost in registers is negligible: ReLU is 1 `max` op; even GELU's tanh approximation (~10 FMAs) is dwarfed by the hundreds of cycles each WGMMA instruction takes.

**CUTLASS 3.x abstraction.** `EpilogueOp` / `EpiOp` stacks pointwise ops (bias → GELU → scale → cast) into the store path. Reference implementation: `cutlass/examples/77_blackwell_dgemm_fusion/dense_gemm_fp8_gelu_persistent.py` fuses **GELU + FP8 GEMM**.

**Our status.** The kernel in this repo is plain (`C = A·B`, no epilogue). Adding fusion is a ~10-line change in the kernel body plus an `activation` constexpr; the host/harness side needs no changes.

---

### 24. Autotuning / Heuristic Dispatch

**Why no single config wins.** The optimal `(BLK_M, BLK_N, BLK_K, NUM_STAGES, SPLIT_K)` is a function:

```
config* = argmax  throughput(M, N, K, dtype, SM_count, smem_per_SM, regs_per_thread)
          subject to:  smem_used ≤ 228 KB   (H20)
                       threads_per_block ≤ 1024
                       regs_per_thread ≤ 255
```

Two competing pressures:

| Pressure | Favors |
|----------|--------|
| Parallelism (small problems) | Small tiles, split-K → more blocks to fill 78 SMs |
| TC efficiency (large problems) | Large tiles → fewer loop iterations, better pipelining |

**Wave-count analysis on H20 (78 SMs).** `waves = ceil(M/BLK_M) × ceil(N/BLK_N) / 78`:

| Problem | Tile 128×256 | Blocks | Waves | SM occupancy |
|---------|--------------|--------|-------|--------------|
| 512³    | 4 × 2        | 8      | 0.10  | 10 % ✗ |
| 1024³   | 8 × 4        | 32     | 0.41  | 41 % ✗ |
| 2048³   | 16 × 8       | 128    | 1.64  | ~100 % (last wave 64 %) |
| 4096³   | 32 × 16      | 512    | 6.56  | ~100 % ✓ |
| 16384³  | 128 × 64     | 8192   | 105   | 100 % ✓ |

Small problems leave SMs idle → trade tile size for parallelism via **split-K**: partition the K-reduction into `sk` splits, each producing a partial `C`, then a reduce kernel sums them.

**PERFLOG-derived dispatch (manual autotuning):**

| Problem | Best variant | Tile / stages | Split-K |
|---------|--------------|----------------|---------|
| 512³    | v7           | 64×64, S5      | sk8 |
| 1024³   | —            | —              | sk2 |
| 2048³   | —            | —              | sk4 |
| 4096³   | v1           | 128×256, S3    | sk4 |
| 16384³  | v1           | 128×256, S3    | sk1 (no split) |

**Runtime dispatch rule** (cheap heuristic, no profiling at call time):

```python
if M * N < 1024 * 1024:
    launch v7  (BLK_M=64,  BLK_N=64,  NUM_STAGES=5)   # small tiles
else:
    launch v1  (BLK_M=128, BLK_N=256, NUM_STAGES=3)   # large tiles
```

**What cuBLAS does.** Ships hundreds of pre-tuned kernels plus a heuristic dispatch table keyed on `(M, N, K, dtype, transposes, SM_count)`. At runtime it picks the closest-matching kernel from the table. This is why cuBLAS looks "impossible to beat" — it is not one kernel, it is a portfolio.

**CUTLASS 3.x equivalent.** `KernelRuntimeFactory` + `Manifest` enumerate the configuration space (tiles, stages, MMA atoms, epilogues) and emit one compiled kernel per combination, then a dispatch table picks among them. Our `PERFLOG/*.md` tables are the **manual** equivalent: sweep configs offline, record TFLOPS, hand-pick winners.

---

### 25. FP8 / Mixed Precision

**FP8 formats on Hopper.**

| Format | Sign | Exp | Mantissa | Range | Precision |
|--------|------|-----|----------|-------|-----------|
| e4m3   | 1    | 4   | 3        | ±448  | ~1 decimal digit (8 values/binade) |
| e5m2   | 1    | 5   | 2        | ±57344| coarser, wider range |
| FP16   | 1    | 5   | 10       | ±65504| ~3 decimal digits |
| BF16   | 1    | 8   | 7        | ±3.4e38 | ~2–3 decimal digits |

WGMMA atom:

```python
op = cute.nvgpu.warpgroup.MmaF8Op(
    cutlass.Float8, cutlass.Float8, cutlass.Float32, instruction_shape=(128, N, 32)
)  # K=32, doubled from FP16's 16
```

**2× throughput mechanism.** Same 16-byte operand fetch from registers, but each element is 8 bits instead of 16 → **2× the FMA operations per WGMMA instruction**. The K dimension of `instruction_shape` doubles (16 → 32) to consume the same register bytes.

| Dtype | H20 peak | Measured (16384³) | % peak |
|-------|----------|--------------------|--------|
| FP16  | 148 TFLOPS | ~139 (cuBLAS)    | 94 % |
| FP8   | 296 TFLOPS | 280.4 TFLOPS      | **94.7 %** |

`Speedup = 296 / 148 = 2.0×`, achieved by changing two lines: the dtype and the MMA atom. Algorithm identical.

**Precision cost.**

| Dtype | Mantissa bits | RE @ 4096³ |
|-------|---------------|------------|
| FP16  | 10            | ~0.01 % |
| FP8 e4m3 | 3          | ~0.14 % |

RE degrades ~14×. Verdict:
- ✅ Inference forward pass — fine (weights and activations quantized, output tolerances loose).
- ❌ Training gradients / optimizer states — FP8 has too little precision for stable gradient accumulation; use FP32 master weights + FP8 activations (mixed precision).

**Shared-memory dividend (that doesn't pay off).** FP8 halves the smem footprint of A/B buffers:

| Dtype | A+B per stage (128×64 + 256×64) | × 3 stages | + C smem (128×256) | Total |
|-------|----------------------------------|------------|---------------------|-------|
| FP16  | 48 KB                            | 144 KB     | 64 KB               | **~214 KB** (≤ 228 KB ✓) |
| FP8   | 24 KB                            | 72 KB      | 32 KB               | **~105 KB** |

FP8 leaves ~123 KB free → could fit 2 blocks/SM (more parallelism) or 6+ pipeline stages. **But:** ncu shows TC utilization already at 90.9 % — the kernel is TC-bound, not smem-bound. Extra smem buys nothing because the bottleneck is the MMA units, not the copy pipeline. The 2× win is purely from the TC doing 2× FMA/cycle on FP8 inputs.

**Algorithmic verdict.** FP8 is a **hardware feature**, not an algorithmic optimization. The 2× comes from the silicon, not from tiling, pipelining, or dispatch work. The user's call: it does not count as a "real breakthrough" — it is free only after NVIDIA built the FP8 tensor cores. Real algorithmic wins (this repo's actual work) are the tiling / pipelining / split-K choices in principles 1–24.

---

### 26. Sparse GEMM (2:4 Structured Sparsity)

Hopper (sm_90) implements **2:4 structured sparsity** in hardware: for every group of 4 contiguous elements along the K-reduction axis, **exactly 2 must be non-zero** and 2 zero (50% sparse). The sparse MMA/WGMMA path skips the zero elements, performing **2× effective FMA/cycle** vs. dense.

**Peak throughput (H20, FP16):**
- Dense peak: `148 TFLOPS` (given)
- Sparse (2:4) peak: `2 × 148 = 296 TFLOPS`
- Theoretical speedup ceiling: `2.0×` (only attainable on fully 2:4-sparse input)

**Metadata cost.** Each group of 4 FP16 elements (8 B) carries a 2-bit selector (which 2 of 4 are non-zero):
- Metadata bytes per row (K elements): `K/4 × 2 bits = K/2 bits = K/16 bytes`
- For `K = 4096`: `256 bytes/row` of indices (vs `8192 bytes/row` of data → 3.1% overhead)
- For a `4096×4096` weight matrix: `4096 × 256 = 1 MiB` metadata, negligible

**When it applies:**
- Weights are **static** and pruned offline → reuses metadata every call (inference sweet spot)
- cuSPARSE `cusparseSpMM` (sparse × dense) on Hopper: typically **1.5–1.9×** over dense `cusparseSpMM` at 50% sparsity
- Training is impractical: gradients have no natural 2:4 structure, pruning-then-retraining (AMPERE/NVIDIA sparse-training recipe) is needed

**Why our kernel cannot benefit (dense random GEMM):**
1. Inputs are i.i.d. Gaussian → effectively 100% non-zero, **no valid 2:4 pattern** exists
2. Forcing 2:4 by zeroing 50% of random data changes the problem — output becomes wrong by up to ~50%
3. Sparse WGMMA uses a **different atom** (`wgmma.sp.mma_async`) + metadata load path; the dense kernel's `cute.nvgpu.warpgroup.MmaF16BF16Op` cannot route through it

**Numbers (hypothetical, 4096³ on H20):**
- Dense cuBLAS: `132.2 TFLOPS` = `89.3%` of 148 T peak (given)
- Sparse cuBLAS, 2:4 weights: up to `~250 TFLOPS` ≈ `84%` of 296 T peak
- Our dense kernel TC util `90.9%` is already near the dense ceiling → sparse is a **different problem class**, not an optimization of this one

**Verdict:** Only applicable with sparse weights (inference). Dense GEMM gains nothing; re-architecting our kernel to sparse WGMMA is out of scope.

---

### 27. Work Stealing (Dynamic Tile Scheduling)

**Static scheduling** assigns each CTA a fixed output tile via `blockIdx`. **Work stealing** replaces this with a global atomic counter `next_tile`; each CTA, after finishing a tile, does `tile_idx = atomicAdd(&next_tile, 1)` and grabs the next available tile. Fast CTAs pick up slack from slow CTAs.

**Imbalance model.** Let `T` = per-tile work, `σ` = std-dev of per-tile time, `S` = SM count, `N` = tile count:
- Static: `t_static ≈ ⌈N/S⌉ · (T_mean + σ_max)` — tail CTA bounded by the slowest tile
- Steal: `t_steal ≈ (N · T_mean / S) + O(σ · log S)` — variance amortizes across SMs
- Gain ≈ `σ / T_mean` fraction; **zero gain when σ = 0**

**Our uniform dense GEMM (σ ≈ 0):**
- Grid for `M=N=4096`, `BLK_M=128, BLK_N=256`: `(32, 16) = 512` tiles / `78` SMs ≈ `6.6` waves
- Every tile does identical `K/BLK_K = 4096/64 = 64` MMA rounds → σ = 0 → **stealing gains ~0%**, only adds atomic overhead (slightly negative)

**Where stealing wins (variance sources):**
| Workload | σ source | Typical gain |
|---|---|---|
| FlashAttention, causal mask | upper-triangle tiles skipped (zero MMA) | **1.3–1.8×** |
| Boundary/residue tiles (M,N not divisible by BLK_M,N) | partial K-reduction on edges | 1.05–1.15× |
| Variable-K / ragged batches | per-tile K differs | 1.2–1.5× |
| Uniform dense (ours) | none | ~1.00× |

**Why our bottleneck is not fixable by stealing.** ncu reports `barrier_stall = 74.3%`, `no_eligible = 96.7%`, `occupancy = 13.9%`. The stall is **intra-tile** (a CTA blocked on its own mbarrier between pipeline stages), not inter-tile. Stealing rebalances *between* tiles but cannot hide a stall *inside* a tile — and with only `13.9%` active occupancy there is no resident warp to switch to. The fix path is deeper pipeline / more occupancy (see §28), not work stealing.

**Implementation blocker in CuTe DSL:**
- `cute.arch.atomic_add` exists, but element-level scalar access on a 0-D gmem tensor is awkward inside the tile abstraction
- Cross-warp broadcast of runtime `tile_idx` requires a runtime `if`/loop the DSL's staged-`if` (`cutlass.const_expr`) cannot express — `const_expr` is compile-time only, so a runtime per-CTA tile counter cannot drive the tile coordinate
- Hence **not implemented**; left as a documented optimization for non-uniform workloads (FlashAttention, ragged GEMM)

**Verdict:** Essential for causal/ragged workloads (~1.3–1.8×). Useless for uniform dense GEMM; our `σ = 0` means stealing only costs atomics.

---

### 28. ncu Profiling Methodology

`ncu` (Nsight Compute) exposes hardware counters that wall-clock timing cannot — it tells you **which pipeline** is saturated and **why** warps stall. The discipline is a 4-step loop: **roofline → Speed-of-Light → stall reasons → hypothesis-fix-verify.**

**Step 1 — Roofline (theoretical ceiling).**
- H20 FP16 peak: `148 TFLOPS`; HBM BW ≈ `4 TB/s`
- Roofline crossover: `AI* = 148e12 / 4e12 = 37 FLOP/byte`
- GEMM arithmetic intensity (FP16, square `N³`): `AI = 2MNK / 2(MK + NK + MN) = MNK / (MK + NK + MN)`; for `M=N=K=4096`: `AI ≈ 4096/3 ≈ 1365 FLOP/byte` ≫ 37 → **compute-bound**, HBM should be idle. Predicts DRAM util low.

**Step 2 — Speed of Light (which pipeline).**
| Metric | Our value | Reading |
|---|---|---|
| `sm__pipe_tensor_op_hmma_cycles_active` (TC) | **90.9%** | near peak — good |
| `dram__throughput` | **7.2%** | HBM idle — matches compute-bound roofline |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | **13.9%** | severely under-occupied |

TC near saturation yet `No Eligible 96.7%` → the SM is idle because **no warp can issue**, not because compute is light. Bottleneck = scheduling/occupancy, hidden behind high TC%.

**Step 3 — Stall taxonomy (`smsp__pcsamp_warps_issue_stalled_*`):**
| Stall reason | Meaning | Our value | Fix lever |
|---|---|---|---|
| `stalled_barrier` | mbarrier / pipeline-stage sync | **74.3%** | fewer syncs / deeper pipeline |
| `stalled_imc_miss` | HBM load wait | low | prefetch, more stages |
| `stalled_long_scoreboard` | smem load wait | — | bank-conflict-free layout |
| `stalled_not_selected` | scheduler picked another warp (healthy) | ~0 | — (occupancy too low) |
| `stalled_short_scoreboard` | arithmetic result wait | — | ILP / instr scheduling |

**Occupancy math (why 13.9%).**
- Threads = `384` → `12 warps/CTA`; Hopper max `64 warps/SM` → resident ceiling `18.75%`
- **smem is the limiter**: per stage `= (128×64 + 256×64)×2 B = 48 KiB`; ×`NUM_STAGES=3` = `144 KiB`. Hopper budget `228 KiB` → **only 1 CTA/SM** fits (`2×144 = 288 > 228`). So `1 CTA × 12 warps = 18.75%` resident, `13.9%` active. The 144 KiB smem *forces* low occupancy.

**Step 4 — hypothesis → fix → verify (the actual experiments):**

| # | Hypothesis | Change | Result | Lesson |
|---|---|---|---|---|
| v4 | fewer barriers → less `stalled_barrier` | `BLK_M=96` (smaller tile, fewer syncs) | **−0.9%** perf | pipeline depth loss > sync savings |
| v7 | deeper pipeline hides barrier | `NUM_STAGES=5` | barrier `74%→8%` ✅ but TC `90.9%→83%` ❌ | 5×48 = 240 KiB > 228 → must shrink tile → TC starves |

The loop explains **why** `BLK_M=128, BLK_N=256, BLK_K=64, S=3` is a local optimum: it sits at the smem-reg-TC tradeoff knee. Fewer stages → barriers dominate; more stages → tile shrinks and TC underutilizes. The number `90.9%` TC is not arbitrary — it's the equilibrium of that tradeoff.

**Cross-check vs. cuBLAS (peak fraction):**
- `512³ → 21.1 T` = `14.3%` of 148 T (launch/grid overhead dominates)
- `4096³ → 132.2 T` = `89.3%` of 148 T ← our kernel competes here (`90.9%` TC)
- `16384³ → 139.2 T` = `94.1%` of 148 T ← remaining headroom = better occupancy/scheduling

**The core loop:** `ncu metric → diagnosis → hypothesis → code change → ncu re-measure → accept/reject`. Numbers drive every decision; `90.9% / 7.2% / 74.3% / 13.9%` is the signature that says "TC-saturated, barrier-bound, occupancy-starved" and rules out HBM-side fixes.
