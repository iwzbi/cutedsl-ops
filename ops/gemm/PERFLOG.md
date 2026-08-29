# GEMM Optimization Journey

Hopper H20 (sm_90, 78 SM, 148 TFLOPS FP16 peak), CuTe DSL (nvidia-cutlass-dsl 4.7.0).

Each step links to the ncu raw report in [`ncu_reports/`](./ncu_reports/).  
Use `git diff v1-baseline..<tag> -- ops/gemm/gemm_kernel.py` to see exact code changes.

---

## Step 1: Baseline — Warp Specialization WGMMA + TMA

**Commit**: `29d7e4a` (tag: `v1-baseline`)

### Architecture
- **MMA**: wgmma.m64n256k16 (warpgroup MMA, A/B from SMEM via descriptor)
- **TMA**: cp.async.bulk.tensor for gmem→smem (A/B load) and smem→gmem (D store)
- **Pipeline**: PipelineTmaAsync, 3-stage double buffering
- **Warp Specialization**: 3 warpgroups (384 threads)
  - WG0 (0-127): Producer — TMA load only, warp0 issues TMA
  - WG1 (128-255): Consumer — WGMMA + epilogue
  - WG2 (256-383): Consumer — WGMMA + epilogue
- **Tile**: BLK_M=128, BLK_N=256, BLK_K=64, NUM_STAGES=3
- **Swizzle**: SW128 (1024-bit swizzle) via hopper_helpers
- **Grid**: (ceil(N/256), ceil(M/128), 1) — N-first

### Performance (CUDA Events, L2-flushed)

| M×N×K | TFLOPS | % peak (148T) | cuBLAS | vs cuBLAS | Waves (78 SM) |
|--------|--------|---------------|--------|-----------|---------------|
| 512³   | 12.1   | 8.2%          | 21.3   | -43%      | 0.03          |
| 1024³  | 51.5   | 34.8%         | 96.9   | -47%      | 0.4           |
| 2048³  | 110.6  | 74.7%         | 127.8  | -13%      | 1.7           |
| 4096³  | 130.6  | 88.2%         | 132.4  | -1.4%     | 6.6           |
| 8192³  | 137.4  | 92.8%         | 137.8  | -0.3%     | 26.3          |
| 16384³ | 141.8  | 95.8%         | 139.3  | **+1.8%** | 211           |

### ncu Profiling (4096³)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Compute (SM) Throughput | 90.9% | TC pipeline near full |
| Memory Throughput | 14.6% | Compute-bound confirmed |
| DRAM Throughput | 7.2% | HBM not bottleneck |
| SM Frequency | 1800 MHz | (boost: 1980) |
| No Eligible Warps | 96.7% | Few warps ready (wgmma long-latency) |
| Cycles/Issued Inst | 67.2 | Each inst waits 67 cycles |
| Top Stall | CTA barrier 74.2% (49.8 cyc) | **Pipeline mbarrier sync** |
| Theoretical Occupancy | 18.75% | smem 214KB + regs 154/thr |
| Achieved Occupancy | 13.9% | Producer idle warps pull down |
| Registers/Thread | 154 | wgmma accumulator heavy |
| L2 Hit Rate | 66.0% | |
| SM Load Imbalance | +6.4%/-8.7% | Grid tail (6.56 waves) |
| Est. Speedup (barrier) | 6.3% | Reduce mbarrier wait |
| Est. Speedup (imbalance) | 5.9% | Split-K for more blocks |

### Bottleneck Analysis
1. **CTA barrier stall (74.2%)**: Pipeline mbarrier sync — consumer_wait waits for TMA data, producer_acquire waits for consumer release. Inherent to pipeline architecture.
2. **SM load imbalance (5.9%)**: 512 blocks / 78 SMs = 6.56 waves, last wave has 44 idle SMs.
3. **Achieved < theoretical occupancy (4.85%)**: Producer warpgroup's 3 non-warp0 threads idle.
4. **Registers (154/thread)**: wgmma accumulator large, limits to 1 block/SM.

### ncu Raw Reports
- [v1-baseline 4096³](ncu_reports/v1-baseline_4096.txt)
- [v1-baseline 1024³](ncu_reports/v1-baseline_1024.txt)

---

## Step 2: Persistent Kernel — CTA Stride Loop (no gain, case study)

**Commit**: (tag: `v2-persistent`)

### What changed
Each CTA loops over multiple output tiles via a stride loop instead of one CTA per tile.
Launch `min(total_tiles, num_sms)` CTAs instead of `total_tiles`.

### Three bugs fixed (discovered via 14-warp-specialization reference)
1. **Deadlock**: `sync_threads()` required all 384 threads, but producer's 128 never reached it → GPU hang. Fix: `pipeline.NamedBarrier(barrier_id=1, num_threads=256)` excluding producer.
2. **>5min compile**: K-tile loop used `unroll_full=True` → 64× code bloat. Fix: `unroll=1`.
3. **Missing register reconfig**: Added `warpgroup_reg_dealloc(40)` (producer) + `warpgroup_reg_alloc(232)` (consumer).

### Performance (no improvement, slightly worse at large scale)

| M×N×K | Baseline | Persistent | Delta |
|--------|----------|------------|-------|
| 1024³  | 51.5     | 51.3       | ~0%   |
| 4096³  | 130.6    | 131.0      | +0.3% |
| 16384³ | 141.8    | 140.9      | -0.6% |

### ncu Comparison (4096³)

| Metric | Baseline | Persistent | Delta |
|--------|----------|------------|-------|
| Compute Throughput | 90.9% | 91.4% | +0.5% |
| Cycles/Issued | 67.2 | 68.7 | +2% worse |
| CTA barrier stall | 74.2% (49.8 cyc) | 74.3% (51.1 cyc) | slightly worse |
| SM load imbalance | est. +5.9% | eliminated | — |

### Why no gain
1. **Tail wave elimination too small**: 4096³ = 6.56 waves → 7 tiles/CTA, imbalance still 6 vs 7.
2. **Per-tile overhead offsets gains**: consumer setup (get_slice/partition/make_fragment) repeated per tile due to DSL staged-if limitation; two NamedBarrier syncs per tile add barrier stall.
3. **Pipeline state continuation**: states persist across tiles (no reset), which works but doesn't reduce stall.
4. **Value of persistent is as prerequisite for epilogue overlap**, not as a standalone optimization — the overhead eats the tail-wave benefit.

### ncu Raw Reports
- [v2-persistent 4096³](ncu_reports/v2-persistent_4096.txt)

---

## Step 3: Block Swizzle — GROUP_M=4 (no gain, L2 worse, case study)

**Commit**: (tag: `v3-swizzle`)

### What changed
Tile assignment uses swizzled (bid_m, bid_n) mapping: tiles are grouped by GROUP_M=4
consecutive M-blocks before striding N, to improve L2 reuse of A tiles.

### Result: L2 hit rate dropped

| Metric | v2-persistent | v3-swizzle | Delta |
|--------|---------------|------------|-------|
| L2 Hit Rate | 66.3% | **62.3%** | -4.0% worse |
| TFLOPS (4096³) | 131.0 | 130.7 | ~0% |
| TFLOPS (16384³) | 140.9 | 141.0 | ~0% |

### Why worse
1. Default M-major order already maximizes A reuse (consecutive M-blocks share A rows)
2. GROUP_M=4 forces 4 M-blocks active simultaneously → 6 active M-blocks per wave vs 5 default
3. 4096³ data = 64MB ≈ L2 60MB → more active blocks = more L2 pressure
4. Kernel is compute-bound (DRAM 7.1%) → L2 optimization is the wrong direction

### ncu Raw Reports
- [v3-swizzle 4096³](ncu_reports/v3-swizzle_4096.txt)

---

## Step 4: BLK_K=96, NUM_STAGES=2 — Fewer Syncs (slightly worse, case study)

**Goal**: Reduce CTA barrier stall by increasing BLK_K (64→96) to halve K-tile count
(64→43 sync cycles), accepting fewer pipeline stages (3→2) to fit smem.

### Result: pipeline depth matters more than sync count

| M×N×K | v2-persistent (K64,S3) | v4-blk96 (K96,S2) | Delta |
|--------|------------------------|---------------------|-------|
| 4096³  | 131.0                  | 129.8               | -0.9% |
| 16384³ | 140.9                  | 140.7               | -0.1% |

### Why worse
1. Pipeline depth 3→2 reduces TMA/WGMMA overlap → more consumer_wait stalls
2. The overlap loss > sync count reduction → net negative
3. BLK_K=96 uses SW64 swizzle (vs SW128 for K=64) — smaller swizzle granularity

### Conclusion
**Pipeline overlap (NUM_STAGES) is more valuable than fewer sync points.**
The 3-stage pipeline hides TMA latency better; the extra sync overhead is offset by overlap.
Can't increase stages (smem full at 209KB/228KB) or reduce syncs (pipeline depth drops).

The CTA barrier stall (74.3%) is **inherent to the TMA+wgmma pipeline architecture**.
The remaining lever is **epilogue overlap** — overlap the epilogue R2S+TMA-store with
the next tile's TMA prefetch, eliminating inter-tile idle time.

---

## Performance Comparison Summary

| Version | 1024³ | 4096³ | 16384³ | TC% | Top Stall | L2 Hit | Occupancy |
|---------|-------|-------|--------|-----|-----------|--------|-----------|
| v1-baseline | 51.5 | 130.6 | 141.8 | 90.9% | CTA barrier 74% | 65.1% | 13.9% |
| v2-persistent | 51.3 | 131.0 | 140.9 | 91.4% | CTA barrier 74% | 66.3% | 13.9% |
| v3-swizzle | — | 130.7 | 141.0 | 91.3% | CTA barrier 74% | 62.3% | 14.1% |
| v4-blk96 | — | 129.8 | 140.7 | — | — | — | — |
