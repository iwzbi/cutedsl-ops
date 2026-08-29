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

## Step 2: Split-K (planned)

**Goal**: Increase grid size for small problems (4096³ only 512 blocks → 6.56 waves).

Split K dimension into multiple partitions, each CTA computes partial result, atomic add to accumulate.

**Expected**: Small-scale +10-20%, large-scale minimal change.

---

## Performance Comparison Summary

| Version | 1024³ | 4096³ | 16384³ | TC% | Top Stall | Occupancy |
|---------|-------|-------|--------|-----|-----------|-----------|
| v1-baseline | 51.5 | 130.6 | 141.8 | 90.9% | CTA barrier 74% | 13.9% |
| v2-splitk | — | — | — | — | — | — |
