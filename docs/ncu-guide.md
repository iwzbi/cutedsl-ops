# Nsight Compute (ncu) 完整使用指南

> 面向使用 CuTe DSL 编写 GPU 算子的开发者。涵盖工具关系、安装、命令语法、
> **所有 `--set full` 报告指标的完整详解**（含含义、性能影响、优化方向、真实示例值）、
> Warp Stall Reasons 全表、Compute Triage 决策树、实战工作流。

---

## 目录

- [Part I: 基础](#part-i-基础)
  - [1. ncu 是什么](#1-ncu-是什么)
  - [2. 安装与权限](#2-安装与权限)
  - [3. 基本用法](#3-基本用法)
  - [4. 关键参数](#4-关键参数)
  - [5. 可用 Section](#5-可用-section)
- [Part II: 完整指标详解](#part-ii-完整指标详解)
  - [6. GPU Speed Of Light Throughput](#6-gpu-speed-of-light-throughput)
  - [7. Compute Workload Analysis](#7-compute-workload-analysis)
  - [8. Memory Workload Analysis](#8-memory-workload-analysis)
  - [9. Scheduler Statistics](#9-scheduler-statistics)
  - [10. Warp State Statistics](#10-warp-state-statistics)
  - [11. Instruction Statistics](#11-instruction-statistics)
  - [12. Launch Statistics](#12-launch-statistics)
  - [13. Occupancy](#13-occupancy)
  - [14. GPU and Memory Workload Distribution](#14-gpu-and-memory-workload-distribution)
  - [15. Source Counters](#15-source-counters)
  - [16. PM Sampling](#16-pm-sampling)
- [Part III: Warp Stall Reasons 全表](#part-iii-warp-stall-reasons-全表)
- [Part IV: Compute Triage 决策树](#part-iv-compute-triage-决策树)
- [Part V: 实战](#part-v-实战)
  - [17. ncu 工作原理](#17-ncu-工作原理)
  - [18. 实战工作流](#18-实战工作流)
  - [19. 常见坑](#19-常见坑)
  - [20. 速查卡片](#20-速查卡片)

---

# Part I: 基础

## 1. ncu 是什么

**ncu** (Nsight Compute) 是 NVIDIA 的 **kernel 级 profiler**。它启动你的程序,
拦截每一个 CUDA kernel launch,用 **kernel replay** 机制（把同一个 kernel 重跑
很多遍，每遍读一组硬件计数器）收集完整的 profiling 数据，然后输出报告。

### 与其他工具的关系

```
CUPTI (底层 C API 库,直接跟 GPU 驱动对话)
  │
  ├──→ Nsight Systems (nsys)     ← 系统级:时间线,「谁在跑,什么时候」
  ├──→ Nsight Compute (ncu)      ← 内核级:硬件计数器,「为什么慢」
  └──→ nvprof (已弃用 sm_80+)    ← 老一代统一工具,被上面两个取代

CUDA Events (独立,不需要 CUPTI)   ← 最简单:只量「跑了多久」

NVTX (标注 API,不是 profiler)     ← 给上面三个工具画标记用
```

| 工具 | 层次 | 回答的问题 | 需要 PMU 权限 |
|---|---|---|---|
| CUDA Events | 计时 | 跑了多久? | 否 |
| nsys | 系统级 | 谁在跑?GPU 何时空转? | 否（活动追踪） |
| ncu | 内核级 | 为什么慢?哪个硬件单元是瓶颈? | **是** |
| nvprof | 混合 | （已弃用 sm_80+） | — |
| NVTX | 标注 | 给 profiler 当标签用 | 否 |

### GPU 硬件模型（理解 ncu 指标的前提）

```
GPU
├── GPC (Graphics Processing Cluster)
│   └── TPC (Texture Processing Cluster)
│       └── SM (Streaming Multiprocessor) ← ncu 大部分指标在这一层
│           ├── SMSP × 4 (sub-partition / warp scheduler)
│           │   ├── Warp Scheduler (每 SMSP 1 个,管理 16 warp pool)
│           │   ├── Register File (每 SMSP 一块)
│           │   ├── Core Pipeline (FP32/INT32/FP64)
│           │   └── Tensor Core (每 SMSP 1 个)
│           ├── Shared Memory / L1 Cache (128KB,可配置分割)
│           └── Unified Data Cache (L1TEX)
├── L2 Cache (所有 SM 共享,H20 = 60MB)
├── DRAM / HBM (全局内存)
└── Memory Controllers (连接 HBM)
```

**关键层次**: `dram__` → `lts__`(L2) → `l1tex__`(L1/TEX/smem) → `smsp__`(SM sub-partition) → `sm__`(SM 汇总)。
ncu 指标名前缀对应硬件层次:`sm__` = SM 级,`smsp__` = sub-partition 级,`l1tex__` = L1/TEX 级,`lts__` = L2 级,`dram__` = DRAM 级。

## 2. 安装与权限

### 安装

ncu 是独立包，不随 CUDA toolkit 一起安装。从 NVIDIA 仓库下载：

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-compute-2026.2.1_2026.2.1.5-1_amd64.deb
sudo dpkg --force-depends -i nsight-compute-*.deb
# 安装到 /opt/nvidia/nsight-compute/2026.2.1/ncu
```

### 权限

ncu 需要 GPU **性能计数器 (PMU)** 权限。在容器/云环境中常被禁用：

```bash
# 检查是否被禁
cat /sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers
# 0 = 所有人可用, 1 = 仅 admin

# 报错信息
# ERR_NVGPUCTRPERM: GPU performance counters are restricted
```

**解决方法**:
- Docker: 启动容器加 `--privileged` 参数
- 物理/虚拟机: 设置 `NVreg_RestrictProfilingToAdminUsers=0` 内核参数

### 版本兼容

ncu 版本必须与 GPU driver 版本匹配。版本不匹配会报 `Failed to prepare kernel`。

## 3. 基本用法

### 命令语法

```bash
ncu [选项] 你的程序 [程序参数]
```

### 标准命令（只看第一个 gemm_kernel launch,full set）

```bash
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
$NCU --set full --target-processes all \
  --kernel-name regex:"gemm_kernel" \
  --launch-skip 0 --launch-count 1 \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096
```

## 4. 关键参数

### Kernel 筛选

| 参数 | 作用 | 常用值 |
|---|---|---|
| `--kernel-name regex:"xxx"` | 只 profile 名字匹配的 kernel | `regex:"gemm_kernel"` |
| `--launch-skip N` | 跳过前 N 个匹配的 launch | `0` |
| `--launch-count C` | 只 profile C 个 launch | `1` |
| `--target-processes all` | 跟踪子进程 | `all` |

简写: `-k` = `--kernel-name`, `-c` = `--launch-count`, `-s` = `--launch-skip`

### Detail Level

```bash
# --set basic(默认):4 个 section,快
#   SpeedOfLight + Occupancy + LaunchStats + WorkloadDistribution
$NCU --set basic -k regex:"gemm_kernel" -c 1 ...

# --set full:所有 section,慢(要 replay ~40 遍)
$NCU --set full -k regex:"gemm_kernel" -c 1 ...

# 也可以挑单个 section
$NCU --section SpeedOfLight --section WarpStateStats \
  -k regex:"gemm_kernel" -c 1 ...
```

### 输出方式

```bash
# 默认:终端文本
$NCU --set full -k regex:"gemm_kernel" -c 1 ...

# 导出文件(GUI 用)
$NCU --set full -k regex:"gemm_kernel" -c 1 \
  --export /tmp/gemm_profile.ncu-rep ...

# CSV(脚本分析用)
$NCU --set full -k regex:"gemm_kernel" -c 1 --csv ... > /tmp/gemm.csv
```

### 查询可用指标

```bash
# 列出所有可用指标(本机 5623 个)
$NCU --query-metrics

# 筛选 tensor core 相关
$NCU --query-metrics 2>&1 | grep -i tensor

# 只看特定指标
$NCU --metrics sm__pipe_tensor_op_hmma_cycles_active.active,\
smsp__pcsamp_warps_issue_stalled_barrier.sum \
  -k regex:"gemm_kernel" -c 1 ...
```

## 5. 可用 Section

| Section | 看什么 | 在 basic set 中? |
|---|---|---|
| `SpeedOfLight` | 峰值 vs 实际吞吐（compute/memory 哪条线满） | ✅ |
| `ComputeWorkloadAnalysis` | 各计算管道利用率（TC/FP32/INT32/FP64） | ❌ |
| `MemoryWorkloadAnalysis` | 各级内存带宽（HBM/L2/L1/smem） | ❌ |
| `SchedulerStats` | warp 调度器状态（有没有 warp ready） | ❌ |
| `WarpStateStats` | stall 原因（为什么 warp 在等） | ❌ |
| `InstructionStats` | 指令类型分布和数量 | ❌ |
| `LaunchStats` | block size / grid / smem / regs | ✅ |
| `Occupancy` | 理论/实际占用率 + 限制因素 | ✅ |
| `WorkloadDistribution` | SM 间负载均衡 | ✅ |
| `SourceCounters` | 按源码行归因的计数器（分支/采样 stall） | ❌ |
| `PmSampling` | 时间线采样（PM 周期采样） | ❌ |
| `Nvlink` | NVLink 利用率 | ❌ |
| `NumaAffinity` | NUMA 亲和性 | ❌ |

---

# Part II: 完整指标详解

> 以下所有指标值来自实测：**H20 GEMM 4096³ fp16, warp specialization
> (3 warpgroups: 1 DMA + 2 MMA), NUM_STAGES=3**, 130.6 TFLOPS (88% peak)。
> 示例值用 `[实测: xxx]` 标注。

## 6. GPU Speed Of Light Throughput

**这个 section 回答**:整体来看,compute 和 memory 哪条线是瓶颈?距离理论峰值有多远?

这是 ncu 最高层级的 summary——先看这里确定大方向。

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **DRAM Frequency** | GHz | 2.62 | HBM 实际运行频率 | 决定 HBM 带宽上限:2×频率×总线宽/8 |
| **SM Frequency** | GHz | 1.80 | SM 实际运行频率 | 决定计算峰值:SMs×FMA/cycle×频率。注意:benchmark 报 1.98 GHz 是 boost,ncu 实测 1.80 是 sustained |
| **Elapsed Cycles** | cycle | 1,910,196 | kernel 总运行周期数 | Duration = Cycles / SM_Freq |
| **Memory Throughput** | % | 14.69 | 整体内存系统吞吐占峰值% | <50% → 不是内存瓶颈;>80% → 内存瓶颈 |
| **DRAM Throughput** | % | 7.23 | HBM 带宽占峰值% | <50% → HBM 远没跑满;>80% → HBM 带宽瓶颈 |
| **Duration** | ms | 1.05 | kernel 实际运行时间 | 与 CUDA Events 测量一致 |
| **L1/TEX Cache Throughput** | % | 15.75 | L1/TEX 缓存管道吞吐 | 包含 smem 访问;高值说明 smem/L1 压力大 |
| **L2 Cache Throughput** | % | 14.01 | L2 缓存吞吐占峰值% | 高值说明 L2 是瓶颈;低值说明数据在 L1/smem 命中 |
| **SM Active Cycles** | cycle | 1,762,599 | SM 实际活跃的周期数 | Active / Elapsed = SM 活跃率(92.4%) |
| **Compute (SM) Throughput** | % | 91.11 | SM 计算吞吐占峰值% | **最重要指标**:>90% = 计算已封顶;<50% = 计算未充分利用 |

### 如何解读

```
Compute Throughput (91%) >> Memory Throughput (15%)  →  compute-bound
```

- **Compute > Memory** → compute-bound（我们的 GEMM 在这）→ 优化计算
- **Memory > Compute** → memory-bound → 优化访存
- **两者都低** → occupancy 不足（SM 空转）→ 增加并行度
- **两者都高** → 接近 optimal

### ncu 建议（INF/OPT 标注）

ncu 在指标表下方给出建议:
- **INF**: 信息性提示（当前状态如何）
- **OPT**: 优化建议 + 估算提速幅度

例如我们的 kernel 触发: `INF: This workload is utilizing greater than 80.0%...` → 表示已接近峰值,进一步优化需要把工作从最忙的单元转移到别的单元。

### SM Active Cycles vs Elapsed Cycles

```
SM Active Cycles / Elapsed Cycles = 1,762,599 / 1,910,196 = 92.4%
```

这 7.6% 的差距 = 部分 SM 在部分时间空转（grid 尾数、负载不均衡）。如果 Active = Elapsed 则所有 SM 全程满载。

### SM Frequency 的陷阱

benchmark 报 1.98 GHz（boost clock）,ncu 实测 1.80 GHz（sustained clock）。这导致 benchmark 按 1.98 算的 peak TFLOPS 偏高。真实 peak = 148 × (1.80/1.98) = 134.5 TFLOPS → 我们 130.6 TFLOPS = 97% of real peak!

### Roofline Chart（GUI only）

ncu GUI 在 Speed of Light 旁边会显示 **Roofline Chart**——kernel 在 roofline 模型上的位置:

```
  TFLOPS
    │
    │         Compute Peak (148T FP16)
    │  ─ ─ ─ ─┬──────────────────────────
    │         │              ● kernel (AI=1365, 130.6T)
    │         │
    │        /│
    │       / │
    │      /  │
    │     /   │
    │    /    │
    │   /     │
    │  /      │
    │ /       │
    │/        │
    ─────────┼────────────────────────── AI (FLOP/Byte)
    0       ~37                        1365
            ↑
         ridge point
         (compute = memory crossover)
```

**怎么读 Roofline Chart**:
- **X 轴**: Arithmetic Intensity (FLOP/Byte) — 每 byte 数据做了多少 FLOP
- **Y 轴**: Achieved TFLOPS — kernel 实际性能
- **斜线** (从原点上升): Memory Bandwidth Roofline — `TFLOPS = AI × HBM_BW`,斜率 = HBM 带宽
- **水平线** (顶部): Compute Peak Roofline — `FP16 peak` / `FP32 peak` / `FP64 peak`
- **ridge point** (两线交点): `AI_ridge = Peak_TFLOPS / HBM_BW`,H20 上 = 148/4.023 ≈ 37 FLOP/Byte
- **kernel 位置**:
  - 在斜线上 → **memory-bound**(被带宽限制,左侧)
  - 在水平线上 → **compute-bound**(被算力限制,右侧)
  - AI < ridge → memory-bound;AI > ridge → compute-bound
  - 在 ridge 附近 → **balanced**(最优位置)

我们的 GEMM:AI = 1365 FLOP/Byte,Achieved = 130.6 TFLOPS → 远在交点右侧 → **compute-bound**(与 Speed of Light 一致)。

CLI 没有图表,但可以从 Speed of Light 的 Compute vs Memory Throughput 判断:
- Compute > Memory → compute-bound(右侧)
- Memory > Compute → memory-bound(左侧)

## 7. Compute Workload Analysis

**这个 section 回答**:SM 里的各计算管道利用率如何?哪条管道最忙?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **Executed Ipc Active** | inst/cycle | 0.13 | 活跃期间每周期执行的指令数 | 极低但可能正常——长延迟指令(wgmma)每条做大量工作 |
| **Executed Ipc Elapsed** | inst/cycle | 0.12 | 整个运行期间每周期执行的指令数 | 比 Active 低说明有空闲周期 |
| **Issue Slots Busy** | % | 3.09 | warp scheduler 发射槽利用率 | <20% = 严重发射饥饿;>60% = 接近容量 |
| **Issued Ipc Active** | inst/cycle | 0.13 | 活跃期间每周期实际发射的指令数 | 与 Executed IPC 通常相等 |
| **SM Busy** | % | 91.11 | SM 管道整体忙碌度 | = Compute Throughput;>90% = 管道已封顶 |

### IPC 低但 SM Busy 高——为什么不矛盾

```
IPC = 0.13 inst/cycle  (极低)
SM Busy = 91.11%       (很高)
```

**不矛盾**。IPC 衡量的是「warp scheduler 发射了多少条指令」,但一条 wgmma 指令占用 tensor core 管道几十个周期。pipe 虽然忙(91%),但指令数少(每 ~8 周期才发一条)。

ncu 原话: "overall pipeline utilization appears to be caused by high-latency instructions"

### 管道层次（从高到低利用率）

ncu 报告中会列出各管道的利用率:

| 管道 | 含义 | 高利用率意味着 |
|---|---|---|
| **Shared** | 共享管(64-bit FP + Tensor) | wgmma 和 FP64 在这;最高通常 = tensor core 主导 |
| **Tensor (Warp Group)** | warp group tensor 管道 | wgmma 指令专用 |
| **FMA** | FMA32 管道 | FP32 运算 |
| **FMA Heavy** | FP64 管道 | FP64 运算 |
| **ALU** | 整数运算管 | 地址计算、索引 |
| **Uniform** | uniform 管道 | warp 统一指令 |

我们的 kernel: Shared 22.8% (最高) → Tensor (FP) 子管主导 → wgmma 是计算主力。

### 优化方向

- **SM Busy > 90%** → 计算封顶,优化空间小
- **某管道 > 80%** → 该管道是瓶颈,考虑转移工作到其他管道
- **IPC < 0.5 且 SM Busy 低** → warp 不足或 stall 严重,看 Warp State Stats
- **Issue Slots Busy < 20%** → 发射饥饿,需要更多 eligible warps

## 8. Memory Workload Analysis

**这个 section 回答**:内存系统各级带宽如何?有没有 spilling?L2 命中率?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **Memory Throughput** | GB/s | 290.89 | 整体内存系统吞吐 | 对比 HBM 带宽峰值 |
| **Mem Busy** | % | 14.69 | 内存系统整体忙碌度 | <50% → 内存不是瓶颈 |
| **Max Bandwidth** | % | 14.42 | 最大带宽利用率 | 通信带宽限制(各级之间) |
| **L1/TEX Hit Rate** | % | 0 | L1/TEX 缓存命中率 | smem 访问不走 L1;0% 是 GEMM 正常(smem 直接访问) |
| **L2 Hit Rate** | % | 66.27 | L2 缓存命中率 | >50% = 大部分数据在 L2;<20% = 频繁穿透到 DRAM |
| **L2 Compression Success Rate** | % | 0 | L2 压缩成功率 | 0% = 数据不可压缩;高重复/零值数据可压缩 |
| **L2 Compression Ratio** | % | 0 | L2 压缩比 | 实际节省的存储比例 |
| **L2 Compression Input Sectors** | sector | 1,050,262 | 尝试压缩的 sector 数 | 压缩量,0% 成功 = 全部浪费了压缩单元时间 |
| **L2 Persisting Size** | MB | 11.80 | L2 持久化缓存大小 | 用 cudaAccessPolicyWindow 设置的持久化区域 |
| **Mem Pipes Busy** | % | 12.15 | 内存管道忙碌度 | 内存发射利用率;<50% → 不是发射瓶颈 |
| **Local Memory Spilling Requests** | — | 0 | 寄存器溢出到 local memory 次数 | >0 = 寄存器不够,溢出到 DRAM,严重影响性能 |
| **Local Memory Spilling Request Overhead** | % | 0 | 溢出开销占比 | >0 = 有性能损失 |
| **Shared Memory Spilling Requests** | — | 0 | smem 溢出请求 | >0 = smem 不够用 |
| **L2 Sector Promotion Misses** | % | 0 | L2 sector 提升失败 | 持久化缓存相关 |

### 如何解读

```
Memory Throughput 14.69% << Compute Throughput 91%  →  compute-bound (确认)
L2 Hit Rate 66%  →  1/3 数据穿透到 DRAM (合理)
L1/TEX Hit Rate 0%  →  正常 (GEMM 数据在 smem,不走 L1)
No Spilling  →  寄存器够用 (154 regs/thread 但没溢出)
```

### L2 Compression（ncu 建议项）

ncu OPT 建议: "Out of 33,608,384 bytes sent to the L2 Compression unit only 0.00% were successfully compressed."

含义: L2 有硬件压缩单元,能压缩含零值或重复值的数据。我们的 GEMM 数据是随机 fp16,不可压缩 → 0% 成功率 → 浪费了压缩单元周期。

优化: 只对含大量零值/重复值的数据标记为可压缩(`cudaMemAdvise` + `cudaMemAccessPolicy`)。GEMM 数据一般不适用。

### 各级内存层次

```
Register (rmem)  ←  最快,0 cycle,每线程私有
    ↓
Shared Memory   ←  ~30 cycle,block 内共享,可 swizzle
    ↓
L1/TEX Cache    ←  ~30 cycle,SM 内共享(与 smem 共用 128KB)
    ↓
L2 Cache        ←  ~200 cycle,全 GPU 共享 (H20 = 60MB)
    ↓
DRAM / HBM      ←  ~400+ cycle,全局内存 (H20 = 96GB HBM3)
```

ncu 的 `l1tex__` 指标看 L1/TEX/smem 层,`lts__` 看 L2 层,`dram__` 看 DRAM 层。

### Memory Tables 子表(GUI + `--set full` CLI)

上面是 summary,ncu `--set full` 还会输出 4 个内存层级的详细子表:

**Shared Memory 子表**:
| 指标 | 含义 |
|---|---|
| Shared Memory Bank Conflict | smem bank 冲突率(bank conflict = 同一 bank 被多 warp 同时访问 → 序列化) |
| Shared Memory Load/Store Throughput | smem 读写带宽 |
| Shared Memory Transactions | smem 访问事务数 |
| Shared Memory Wave Pool | smem 访问的 wave 池化(冲突导致的多 wave) |

**L1/TEX Cache 子表**:
| 指标 | 含义 |
|---|---|
| L1/TEX Hit Rate | L1 命中率(GEMM 通常 0%,数据在 smem) |
| L1/TEX Cache Misses | L1 未命中数 |
| L1/TEX Throughput | L1 吞吐 |
| L1/TEX Cache Sector Miss Ratio | 每 sector miss 比率(一个 sector = 32B) |
| L1/TEX Cache Sector Hit Ratio | sector 命中比 |

**L2 Cache 子表**:
| 指标 | 含义 |
|---|---|
| L2 Hit Rate | L2 命中率(>50% = 大部分数据在 L2) |
| L2 Throughput | L2 吞吐 |
| L2 Sector Requests | L2 sector 请求数 |
| L2 Compression Success Rate | L2 压缩成功率(零值/重复数据可压缩) |
| L2 Eviction Policy | L2 驱逐策略(LRU/stream/persisting) |
| L2 Persisting Hit Rate | persisting 区域命中率 |

**Device Memory (DRAM) 子表**:
| 指标 | 含义 |
|---|---|
| DRAM Throughput | HBM 实际带宽 |
| DRAM Bytes Read/Written | 读写字节数 |
| L2 to DRAM Throughput | L2→DRAM 回写带宽 |
| DRAM to L2 Throughput | DRAM→L2 加载带宽 |

**怎么用**:如果 Memory Throughput summary 显示内存是瓶颈(>50%),看这 4 个子表确定是哪一级:
- Shared Memory bank conflict 高 → swizzle 不对或访问模式有冲突
- L1 miss 多 → 增加 L1 命中(调整 tile 大小)
- L2 hit rate 低 → 数据太大,考虑 tiling 或 L2 persisting
- DRAM throughput 高 → 真的是 HBM 带宽瓶颈

## 9. Scheduler Statistics

**这个 section 回答**:warp scheduler 有多少 warp 可以发射?延迟隐藏得够不够?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **No Eligible** | % | 96.67 | 没有任何 warp 可发射的周期占比 | **关键指标**:高 = 延迟隐藏不足 |
| **One or More Eligible** | % | 3.33 | 至少有一个 warp 可发射的周期占比 | = 100% - No Eligible |
| **Active Warps Per Scheduler** | warp | 2.24 | 每个 scheduler 平均活跃 warp 数 | 对比最大 16;低 = occupancy 低 |
| **Eligible Warps Per Scheduler** | warp | 0.04 | 每个 scheduler 平均可发射 warp 数 | <<1 = 几乎没 warp ready;这是 No Eligible 96.67% 的根因 |
| **Issued Warp Per Scheduler** | warp | 0.03 | 每个 scheduler 平均实际发射 warp 数 | 每 ~30 周期发一条指令 |

### Warp 生命周期

```
Theoretical Warps (最大 16/scheduler) → Active Warps (实际驻留)
  → Eligible Warps (not stalled,可发射)
    → Issued Warp (被 scheduler 选中的那个)
```

每周期:
1. scheduler 检查所有 active warps 的状态
2. 没在 stall 的 = eligible
3. 从 eligible 中选 1 个 = issued
4. 如果没有 eligible → No Eligible → 空过这一周期

### 96.67% No Eligible 但 91% SM Busy——为什么不矛盾

```
No Eligible: 96.67%  (几乎没 warp ready)
SM Busy:     91.11%  (SM 管道几乎满载)
```

**不矛盾**! wgmma 是**异步长延迟指令**:warp 发出 wgmma 后就 stall(等结果或等 mbarrier),但 **tensor core 管道在持续工作**。scheduler 没有 eligible warp,但管道在忙——这正是 Hopper 的设计:用长时指令填满管道,不靠多 warp 隐藏延迟。

如果 IPC 高 + No Eligible 低 → 传统短延迟指令的延迟隐藏
如果 IPC 低 + No Eligible 高 + SM Busy 高 → 长延迟指令主导(wgmma 场景,正常)

### ncu 建议

ncu OPT: "Est. Local Speedup: 8.888% — reduce the time the active warps are stalled by inspecting the top stall reasons"

→ 减少 stall 可以让更多 warp eligible → 更多指令发射 → 性能提升。但注意:在 wgmma 场景下,stall 是固有的(等 mbarrier 同步),不可能完全消除。

## 10. Warp State Statistics

**这个 section 回答**:warp 每条指令之间在等什么?哪个 stall 原因占比最大?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **Warp Cycles Per Issued Instruction** | cycle | 67.16 | 每条指令之间 warp 等了多少周期 | **核心指标**:越高 = 需要越多 warp 并行来隐藏延迟 |
| **Warp Cycles Per Executed Instruction** | cycle | 67.16 | 同上(用 executed 而非 issued) | 通常与 Issued 相等 |
| **Avg. Active Threads Per Warp** | — | 31.98 | 每条指令平均活跃线程数 | 接近 32 = 无 divergence;<24 = 有分支发散 |
| **Avg. Not Predicated Off Threads Per Warp** | — | 29.34 | 每条指令实际执行(非 predicated off)的线程数 | 低于 Active Threads = predication 开销 |

### Warp Cycles Per Issued Instruction = 67 的含义

每发出一条指令,warp 平均等 67 个周期才能发下一条。在这 67 个周期里:

```
49.8 cycles  stalled at CTA barrier (74.2%)
  + 一些 cycles 在等其他原因
  = 67.2 cycles total
```

→ 74.2% 的等待时间花在 CTA barrier（mbarrier 同步）上。

### Avg. Active Threads Per Warp = 31.98

接近 32（warp 大小）= 几乎无 divergence。我们的 kernel 所有线程走相同路径（producer/consumer 分支按 warpgroup 整齐切分,不造成 warp 内 divergence）。

### ncu 建议（stall 原因详解）

ncu OPT: "On average, each warp spends 49.8 cycles being stalled waiting for sibling warps at a CTA barrier."

→ **CTA barrier = pipeline mbarrier 同步**。consumer_wait 等 TMA 数据到达,producer_acquire 等 consumer 释放。这是 warp specialization 的固有代价。

完整 stall 原因表见 [Part III: Warp Stall Reasons 全表](#part-iii-warp-stall-reasons-全表)。

### Stall Reason 分布图表(GUI + CLI)

上面的 4 个 summary 指标是**汇总**,真正的诊断价值在 stall reason 分布图表里。ncu 在 Warp State Statistics 下方会列出**每个 stall 原因的 cycles per issued instruction**:

```
Stall Reason                              Cycles/Issued Inst   占比
stalled_barrier (CTA Barrier)                    49.80          74.2%
stalled_long_scoreboard (Long Scoreboard)          5.21           7.8%
stalled_not_selected (Not Selected)                4.31           6.4%
stalled_imc_miss (Instruction Cache Miss)          2.15           3.2%
stalled_math_pipe_throttle                         1.87           2.8%
stalled_short_scoreboard                            1.12           1.7%
stalled_drain                                       0.92           1.4%
stalled_misc                                        0.62           0.9%
stalled_tex                                         0.41           0.6%
stalled_no_instruction                              0.31           0.5%
stalled_lg_throttle                                 0.18           0.3%
stalled_mio_throttle                                0.08           0.1%
stalled_membar                                      0.05           0.1%
stalled_warpgroup_arrive                            0.00           0.0%
stalled_math_pipe_full                              0.00           0.0%
stalled_wait                                        0.00           0.0%
────────────────────────────────────────────────
Total                                     67.16         100.0%
```

**怎么读这个表**:
1. **找 top-1 stall reason** → 49.8 cycles at CTA barrier = 74.2% → 这就是主要瓶颈
2. **看占比 >10% 的** → 这些是值得优化的;<5% 的通常不值得单独处理
3. **所有 stall reason 的 cycles 加起来** = Warp Cycles Per Issued Instruction (67.16)
4. **每个 stall 原因的优化方向**见 [Part III](#part-iii-warp-stall-reasons-全表)

CLI 输出中这个表紧跟在 Warp State Statistics summary 下方。GUI 中以柱状图显示,更直观。

## 11. Instruction Statistics

**这个 section 回答**:kernel 执行了多少条指令?发射效率和执行效率差多少?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **Executed Instructions** | inst | 18,203,648 | 执行的 warp 指令总数 | 每条指令 = 1 个 warp 执行 |
| **Issued Instructions** | inst | 18,204,662 | 发射的 warp 指令总数 | >Executed = 有重发(no-op/predicated off) |
| **Avg. Executed Instructions Per Scheduler** | inst | 58,345 | 每 scheduler 平均执行指令数 | 总量 / 4 scheduler × SMs |
| **Avg. Issued Instructions Per Scheduler** | inst | 58,348 | 每 scheduler 平均发射指令数 | Issued - Executed = ~1 = 几乎无重发 |
| **Local Memory Spilling Requests** | byte | 0 | 寄存器溢出量 | 0 = 无溢出 |
| **Shared Memory Spilling Requests** | byte | 0 | smem 溢出量 | 0 = 无溢出 |

### Issued vs Executed

- **Executed** = warp 实际执行了这条指令（可能 predicated off 部分线程）
- **Issued** = warp 被选中发射了（包括 no-op、predicated off 全部线程的情况）
- Issued > Executed = 有指令被发射但没实际执行（浪费 issue slot）

我们的: Issued - Executed = 18,204,662 - 18,203,648 = 1,014 → 几乎无浪费。

### Instruction Mix Chart（GUI only）

ncu GUI 在 Instruction Statistics 旁边会显示 **Instruction Mix Chart**——kernel 执行的指令按类型分布:

```
指令类型                  占比
FMA/MMA (tensor core)     ████████████████  85%  ← wgmma 占绝大多数
LSU (load/store)          ██                  8%  ← smem/gmem 访存
ALU (int/float)           █                   3%  ← 地址计算
Branch                    ▏                    0.05%
Other                     ▏                    4%
```

**怎么用**:
- **MMA 占比高** → 计算密集型 kernel(GEMM 正常)
- **LSU 占比高** → 访存密集型(memory-bound 可能)
- **ALU 占比高** → 地址计算/逻辑开销大,可能可以简化 index 计算
- **Branch 占比高** → 分支多,可能有 divergence

对 WGMMA kernel 特别注意:**wgmma 是一条指令但做大量计算**,所以 IPC 低(0.13)不等于性能差——关键是 tensor core 管道利用率(Speed of Light Compute Throughput)。

## 12. Launch Statistics

**这个 section 回答**:kernel 的启动配置是什么?grid/block/threads/regs/smem 各多少?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **Block Size** | — | 384 | 每 block 线程数 = 3 warpgroups × 128 | 与 occupancy 相关 |
| **Grid Size** | — | 512 | 总 block 数 | Grid/SMs = waves;少 = 欠载 |
| **Registers Per Thread** | reg/thread | 154 | 每线程寄存器数 | **关键**:影响 occupancy;154×384=59,136/65,536 → 1 block/SM |
| **Shared Memory Configuration Size** | KB | 233.47 | 每 block smem 总量 | 含 dynamic + driver + static |
| **Driver Shared Memory Per Block** | KB/block | 1.02 | 驱动占用 smem | 不可控 |
| **Dynamic Shared Memory Per Block** | KB/block | 214.02 | 动态分配 smem | 我们用 SmemAllocator 分配的 |
| **Static Shared Memory Per Block** | byte/block | 0 | 静态 smem | `__shared__` 变量,我们没用 |
| **Stack Size** | — | 1024 | 每 thread 栈大小 | 递归/局部变量 |
| **Threads** | thread | 196,608 | 总线程数 = Block × Grid | = 384 × 512 |
| **# SMs** | SM | 78 | GPU SM 数 | H20 = 78 |
| **# TPCs** | — | 39 | TPC 数 | 每 TPC 含 2 SM |
| **Waves Per SM** | — | 6.56 | Grid / SMs = 512/78 | <1 = 欠载;>1 = 多波 |
| **Cluster Size** | — | 0 | Cluster 大小 | 0 = 不用 cluster |
| **Cluster Scheduling Policy** | — | PolicySpread | Cluster 调度策略 | 无 cluster 时无意义 |
| **Function Cache Configuration** | — | CachePreferNone | L1/smem 分配偏好 | 默认;可设 CachePreferShared/CachePreferL1 |
| **Enabled TPC IDs** | — | all | 启用的 TPC | 可能因 MIG 禁用部分 |
| **Uses Green Context** | — | 0 | 是否用 Green Context | 通常 0 |

### Registers Per Thread = 154 的影响

```
154 regs/thread × 384 threads = 59,136 registers
H20 每 SM 最大 65,536 registers
59,136 / 65,536 = 90.2% → 只能放 1 个 block/SM
```

wgmma 累加器(128×256 fp32 = 大量寄存器)是 154 regs/thread 的主因。减小 tile 可以降 regs 但改变算法。

### Waves Per SM = 6.56 的含义

```
512 blocks / 78 SMs = 6.56 waves
→ 6 个完整波(每波 78 SMs 满载) + 0.56 波(44 SMs 空闲)
→ 最后一波有 34 个 SM 空转 = 5.9% 性能损失
```

### Shared Memory 总量

```
Dynamic (214.02 KB) + Driver (1.02 KB) + Static (0) = 233.47 KB per block
H20 每 SM 最大 228 KB smem
→ 233.47 > 228? 不,Dynamic 214 KB < 228 KB,加上 driver 1 KB = 215 KB 可用
→ ncu 报 233.47 含了一些 overhead
→ 只能放 1 block/SM (smem-limited)
```

## 13. Occupancy

**这个 section 回答**:理论占用率和实际占用率各多少?什么资源限制了占用率?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **Theoretical Occupancy** | % | 18.75 | 硬件限制下的最大占用率 | = Theoretical Warps / Max Warps |
| **Achieved Occupancy** | % | 13.90 | 实际测到的占用率 | <Theoretical = 有 warps 没驻留 |
| **Theoretical Active Warps per SM** | warp | 12 | 理论最大活跃 warp 数 | = 18.75% × 64 |
| **Achieved Active Warps Per SM** | warp | 8.89 | 实际活跃 warp 数 | <Theoretical = 不均衡 |
| **Block Limit Registers** | block | 1 | 寄存器限制的 block 上限 | 154 regs → 只能 1 block |
| **Block Limit Shared Mem** | block | 1 | smem 限制的 block 上限 | 214KB → 只能 1 block |
| **Block Limit Warps** | block | 5 | warp 槽限制的 block 上限 | 384 threads = 12 warps;64/12≈5 blocks |
| **Block Limit Barriers** | block | 32 | barrier 限制的 block 上限 | 每 SM 32 barriers |
| **Block Limit SM** | block | 32 | SM slot 限制 | 每 SM 最多 32 blocks |
| **Max Active Clusters** | cluster | 0 | 最大活跃 cluster | 不用 cluster = 0 |
| **Max Cluster Size** | block | 12 | 最大 cluster 大小 | 每 SM 12 blocks |
| **Overall GPU Occupancy** | % | 0 | 全 GPU 占用率 | 不确定为何 0 |
| **Cluster Occupancy** | % | 0 | cluster 占用率 | 不用 cluster = 0 |

### 占用率限制链

```
Block Limit = min(
    Block Limit Registers (1),   ← 154 regs/thread → min!
    Block Limit Shared Mem (1),  ← 214 KB → min!
    Block Limit Warps (5),
    Block Limit Barriers (32),
    Block Limit SM (32)
) = 1 block/SM

Theoretical Warps = 1 block × 12 warps/block = 12 warps
Theoretical Occupancy = 12 / 64 = 18.75%
```

**两个限制因素同时 = 1**:registers 和 smem。要增加 occupancy,必须同时降低两者。

### Achieved (13.9%) < Theoretical (18.75%)

```
Achieved Active Warps = 8.89
Theoretical Active Warps = 12
```

差距原因:
1. **Producer warpgroup 的 3 个非 warp0 线程 idle** — 只有 warp0 发 TMA,warp1/2/3 大部分时间不活跃
2. **Grid 尾数** — 最后一波不满 SM,部分 SM 没有活跃 block
3. **启动/结束阶段** — kernel 启动和结束时 occupancy 低于稳态

### 低 Occupancy 但高性能——Hopper 哲学

```
Theoretical Occupancy: 18.75%  (极低)
Achieved Occupancy:    13.90%  (更低)
SM Busy:               91.11%  (很高!)
```

传统 GPU:高 occupancy → 多 warp → 隐藏延迟 → 高性能
Hopper:长时 wgmma 指令 + TMA 流水线 → 不需要多 warp → 低 occupancy 也高性能

ncu 建议: "The 3.00 theoretical warps per scheduler are below the hardware maximum of 16. This kernel's theoretical occupancy is limited by registers and shared memory."

→ 可以降低 regs/smem 来增加 occupancy,但当前已经 91% SM Busy,增加 occupancy 不一定提升性能。

## 14. GPU and Memory Workload Distribution

**这个 section 回答**:SM 之间负载均衡吗?有没有某些 SM 特别闲?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **Average SM Active Cycles** | cycle | 1,762,599 | SM 平均活跃周期 | 对比 Total = 利用率 |
| **Total SM Elapsed Cycles** | cycle | 147,311,104 | SM 总周期 = Elapsed × SMs | = 1,910,196 × 78 (但 ncu 取的是 147M ≈ 1.88M × 78) |
| **Average SMSP Active Cycles** | cycle | 1,751,263 | sub-partition 平均活跃周期 | 略低于 SM(some SMSP 空闲) |
| **Total SMSP Elapsed Cycles** | cycle | 589,244,416 | sub-partition 总周期 | = Total SM × 4 |
| **Average DRAM Active Cycles** | cycle | 199,224 | DRAM 平均活跃周期 | 低 = HBM 不是瓶颈 |
| **Total DRAM Elapsed Cycles** | cycle | 132,240,896 | DRAM 总周期 | 按 DRAM 频率算 |
| **Average L1 Active Cycles** | cycle | 1,762,599 | L1 平均活跃周期 | 与 SM 相同(L1 在 SM 内) |
| **Total L1 Elapsed Cycles** | cycle | 147,311,104 | L1 总周期 | |
| **Average L2 Active Cycles** | cycle | 1,322,745 | L2 平均活跃周期 | 低于 SM → L2 不是瓶颈 |
| **Total L2 Elapsed Cycles** | cycle | 171,948,768 | L2 总周期 | |

### SM Load Imbalance

ncu OPT: "One or more SMs have a much lower number of active cycles than the average. Maximum instance value is 6.35% above the average, while the minimum instance value is 8.75% below the average."

```
max: +6.35% above average
min: -8.75% below average
差值: 15.1%
Est. Speedup: 5.925%
```

原因: 512 blocks / 78 SMs = 6.56 waves → 最后一波 44 SMs 没活干 → 这 44 个 SM 比平均少 ~8.75% 活跃周期。

优化: Split-K 增加 blocks 数;或调整 tile size 让 grid 整除 SM 数。

### SMSP Load Imbalance

与 SM 类似但更细粒度(sub-partition 级)。差值相近(6.37% / 8.80%)。

## 15. Source Counters

**这个 section 回答**:分支效率如何?有没有 warp divergence?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **Branch Instructions Ratio** | % | 0.05 | 分支指令占总指令比 | 极低 = 计算密集型 kernel |
| **Branch Instructions** | inst | 940,544 | 分支指令总数 | |
| **Branch Efficiency** | % | 100 | 分支预测命中率 | 100% = 完美(warp 内无 divergence) |
| **Avg. Divergent Branches** | branches | 0 | 平均发散分支数 | 0 = 无 warp divergence |

### Branch Efficiency = 100%

```
Branch Efficiency = (Total Branches - Divergent Branches) / Total Branches × 100
= (940,544 - 0) / 940,544 × 100 = 100%
```

我们的 kernel 的分支(producer/consumer if/else)是按 warpgroup 整齐切分的——同一 warp 内所有线程走同一路径 → 无 divergence。

### Source Counters 还有什么(GUI only)

在 GUI 中,Source Counters 还会显示:

**1. Warp Stall Sampling（最重要的源码归因工具）**

ncu 按源码行归因 stall 原因——告诉你**哪一行代码导致了最多的 stall**:

```
源码行          Stall 原因              Stall 样本数    占比
gemm_kernel.py:265  stalled_barrier          12,847       42%   ← consumer_wait
gemm_kernel.py:278  stalled_long_scoreboard   3,215       11%   ← wgmma 等结果
gemm_kernel.py:301  stalled_barrier           2,104        7%   ← producer_acquire
gemm_kernel.py:295  stalled_not_selected      1,873        6%   ← scheduler 空转
...
```

**怎么用**:
- 找 stall 样本最多的行 → 那行就是优化目标
- `warpsampling:` 前缀的指标就是这个(Sampling-based,不是精确计数)
- CLI 需要编译带 `-g`(debug 信息)+ `--lineinfo`(CUDA),CuTe DSL 编译的 kernel 默认带 `--generate-line-info`

**2. Excessive Metrics**
- 某些行的访存效率差(wavefronts/sectors/requests 过多)
- 例:某行每个 warp 发了 4 个 sector 请求但只用 1 个 → 75% 浪费

**3. Instructions Executed**
- 按行的指令数(热点行)
- 找执行最多的行 → 内层循环体

CLI 的 `--set full` 只显示汇总,不按行归因(需要 debug 信息 + GUI)。

## 16. PM Sampling

**这个 section 回答**:kernel 运行期间,指标随时间变化吗?

| 指标 | 单位 | 实测值 | 含义 | 性能影响 |
|---|---|---|---|---|
| **Maximum Buffer Size** | MB | 42.21 | PM 采样缓冲区大小 | 决定能采多少样本 |
| **Maximum Sampling Interval** | µs | 1.50 | 最大采样间隔 | 间隔短 = 数据密但开销大 |
| **# Pass Groups** | — | 2 | 采样分组数 | 不同指标组需要不同 pass |

PM Sampling 是 ncu 2026.2 的新特性:不再只看 kernel 平均值,而是看随时间的**变化曲线**。在 GUI 中以时间线显示,CLI 只有汇总。

用途: 看 kernel 运行中是否有阶段性行为变化(如 prefetch 阶段 vs mainloop 阶段 vs epilogue 阶段)。

---

# Part III: Warp Stall Reasons 全表

> 当 warp 不能发射指令时,stall 原因告诉你为什么。以下按类别分组,
> 每个含:含义、常见场景、优化方向。
>
> **注意**: stall 出现在**依赖指令**上,不是导致 stall 的指令本身。
> 例如:慢 load 不在 load 指令上报 stall,而是在使用 load 结果的指令上报 stall;
> barrier 在 barrier 后面的指令上报 stall,不是 barrier 本身。

## 同步类 Stall

### `stalled_barrier` (CTA Barrier)

- **含义**: warp 在等 CTA barrier(`__syncthreads()` 或 mbarrier)
- **我们的实测**: 49.8 cycles / 67.2 total = **74.2%** (最大 stall 原因)
- **常见场景**: pipeline mbarrier 同步(consumer_wait / producer_acquire)、`__syncthreads()`、cooperative groups sync
- **优化方向**:
  - 均衡 barrier 前的工作量(避免某些 warp 先到等着)
  - 减少 barrier 频率(增大 K-tile,减少 sync 次数)
  - Epilogue overlap(把 epilogue 和下一 tile 的 TMA 重叠)
  - 如果 block ≥ 512 threads,考虑拆小

### `stalled_membar` (Memory Fence)

- **含义**: warp 在等 memory fence(`__threadfence()` 等)
- **常见场景**: `__threadfence()` / `__threadfence_block()` / `__threadfence_system()`
- **优化方向**: 减少 fence 范围和频率;用 lighter fence 替代

### `stalled_drain` (Drain)

- **含义**: warp 在等所有未完成操作 drain(排空)
- **常见场景**: kernel 结束前的清理、大 bar 后的 drain
- **优化方向**: 通常不可控(kernel 结尾固有)

### `stalled_wait` (Wait)

- **含义**: warp 在等 fixup 或 transfer 操作完成
- **常见场景**: 特殊指令的 fixup 阶段
- **优化方向**: 通常不可控

## 内存类 Stall

### `stalled_long_scoreboard` (Long Scoreboard)

- **含义**: warp 在等内存依赖(L1TEX: local/global/surface/tex load 的结果)
- **常见场景**:
  - smem load 没回来(cute.copy g2s 的结果在使用时还没到)
  - global load 没回来
  - bank conflict 延长了 smem 访问
- **优化方向**:
  - 增加 ILP(更多独立指令填充等待间隙)
  - 检查 bank conflict(`l1tex__data_bank_conflicts_pipe_lsu_mem_shared`)
  - 预取(prefetch 更早发出 load)
  - 检查 coalescing(sector/request 是否理想)

### `stalled_short_scoreboard` (Short Scoreboard)

- **含义**: warp 在等 MIO(Memory I/O)依赖,excluding L1TEX
- **常见场景**:
  - smem store 的结果在后续 read 时还没完成
  - 特殊寄存器(CAR)操作
- **优化方向**:
  - 检查 bank conflict(与 long_scoreboard 互补:short = smem store→read, long = smem/global load)
  - 减少 smem store→read 的依赖链

### `stalled_imc_miss` (Instruction/Memory Cache Miss)

- **含义**: warp 在等 L2→DRAM 的 miss(L2 未命中,要等 HBM)
- **常见场景**: 数据不在 L2,穿透到 HBM(~400 cycle 延迟)
- **优化方向**:
  - 增大数据局部性(让更多访问命中 L2)
  - 增加并行 load(更多 bytes in flight 隐藏 HBM 延迟)
  - 使用 L2 persisting cache(`cudaAccessPolicyWindow`)

### `stalled_lg_throttle` (LSU Throttle)

- **含义**: L1 指令队列满了,load/store 单元反压
- **常见场景**: 太多 load/store 指令同时 in-flight,L1 队列饱和
- **优化方向**: 减少 in-flight memory ops;增大每条 op 的粒度(向量化)

### `stalled_mio_throttle` (MIO Throttle)

- **含义**: MIO(Memory I/O)管道饱和
- **常见场景**: smem + global 混合访问太多
- **优化方向**: 减少 memory op 数量

## 调度类 Stall

### `stalled_not_selected` (Not Selected)

- **含义**: warp eligible(可发射)但 scheduler 没选它(选了别的 warp)
- **常见场景**: 正常!有多个 eligible warp,scheduler 只能选一个
- **注意**: **不是真正的 stall**,是正常调度行为。高 = 有足够的 warp 并行(好事)
- **优化方向**: 不需要优化

### `stalled_warpgroup_arrive` (Warpgroup Arrive)

- **含义**: warp 在等 warpgroup 内其他 warp 到达同步点(wgmma 的跨 scheduler 同步)
- **常见场景**: wgmma 指令——4 个 scheduler 的 warp 必须都到达才能执行
- **优化方向**: 均衡 4 个 scheduler 的工作量(避免某个 scheduler 的 warp 迟到)

### `stalled_math_pipe_throttle` (Math Pipe Throttle)

- **含义**: 计算管道满了(FP32/FP64/INT 管道饱和)
- **常见场景**: 大量同类型计算指令
- **优化方向**: 混合指令类型(用不同管道的指令交替)

### `stalled_math_pipe_full` (Math Pipe Full)

- **含义**: 计算管道队列满
- **常见场景**: 连续大量同类型计算指令
- **优化方向**: 增加指令多样性

### `stalled_tex` (Texture)

- **含义**: 纹理/surface 操作 stall
- **常见场景**: 纹理内存访问
- **优化方向**: 一般 GEMM 不涉及

### `stalled_misc` (Miscellaneous)

- **含义**: 其他未分类原因
- **常见场景**: 少见,通常 <1%
- **优化方向**: 忽略

### `stalled_no_instruction` (No Instruction)

- **含义**: 指令 cache miss,没有可发射的指令
- **常见场景**: kernel 太大或 icache 容量不够
- **优化方向**: 减小 kernel 代码量

---

# Part IV: Compute Triage 决策树

> 从 ncu 数据到优化方向的系统决策流程（基于 NVIDIA Compute Triage Guide）。

## 决策 1: kernel 是否接近峰值?

```
SpeedOfLight: Compute (SM) Throughput
  ├─ > 80%  → 已接近峰值,优化空间有限
  │   └─ 看 Workload Distribution: SM 是否均衡?
  │       ├─ 均衡 → 看 Decision 4/5 (具体管道/内存瓶颈)
  │       └─ 不均衡 → grid 尾数问题 → Split-K / 调整 tile
  ├─ 50-80% → 有优化空间
  │   └─ 看 Scheduler: No Eligible 高吗?
  │       ├─ 是 → 延迟隐藏不足 → Decision 3
  │       └─ 否 → 可能 occupancy 低 → Decision 2
  └─ < 50% → 严重未利用
      └─ 看 Occupancy: Theoretical 低吗?
          ├─ 是 → Decision 2 (降 regs/smem/threads)
          └─ 否 → 均衡/调度问题 → Workload Distribution
```

## 决策 2: Occupancy 不足怎么办?

```
Block Limit = min(Registers, SharedMem, Warps, Barriers, SM)
  ├─ Registers limited (我们的情况: 154 regs → 1 block)
  │   ├─ 减少 regs: --maxrregcount / __launch_bounds__
  │   ├─ 减小 accumulator tile (BLK_M/N)
  │   └─ 注意: 减 regs 可能增 spilling → 看 InstructionStats
  ├─ SharedMem limited (我们的情况: 214KB → 1 block)
  │   ├─ 减小 tile 或 NUM_STAGES
  │   ├─ 减小 BLK_M/N/K
  │   └─ 注意: 可能影响算法正确性
  ├─ Warps limited
  │   └─ 用更小的 block size
  └─ SM/Barriers limited
      └─ 调整 maxThreadsPerBlock
```

## 决策 3: 延迟隐藏不足(No Eligible 高)怎么办?

```
No Eligible > 50% + IPC < 1.0
  ├─ 看 Warp State Stats: top stall reason
  │   ├─ barrier (我们的情况: 74.2%)
  │   │   ├─ 均衡 barrier 前的工作量
  │   │   ├─ 减少 sync 频率(增大 tile)
  │   │   └─ epilogue overlap
  │   ├─ long_scoreboard (内存依赖)
  │   │   ├─ 检查 bank conflict
  │   │   ├<arg_value> 增加 ILP (独立指令填充等待)
  │   │   ├─ prefetch 更早
  │   │   └─ 检查 coalescing
  │   ├─ imc_miss (HBM miss)
  │   │   ├─ 增大 L2 命中率(数据局部性)
  │   │   ├─ 增加并行 load
  │   │   └─ L2 persisting cache
  │   ├─ not_selected (正常,不需优化)
  │   ├─ lg_throttle (L1 队列满)
  │   │   └─ 减少 in-flight memory ops
  │   └─ math_pipe_throttle/full
  │       └─ 混合指令类型
  └─ 检查 Occupancy: Theoretical 低?
      ├─ 是 → Decision 2 (增加 occupancy)
      └─ 否 → stall 是主因,不是 occupancy
```

## 决策 4: 计算瓶颈在哪个管道?

```
Compute Workload Analysis: 哪个管道利用率最高?
  ├─ Shared (Tensor FP) → wgmma/tensor core 主导
  │   └─ 看是否有非计算工作能转移出去
  ├─ FMA → FP32 主导
  │   └─ 考虑用 tensor core 替代 FP32 运算
  ├─ ALU → 整数/地址计算主导
  │   └─ 优化索引计算,考虑预计算
  ├─ FP64 → FP64 主导
  │   └─ 考虑用 FP32 + 折后校正
  └─ Uniform → uniform 指令
      └─ 减少 uniform 操作
```

## 决策 5: 内存瓶颈在哪一级?

```
MemoryWorkloadAnalysis: 哪级内存 throughput 最高?
  ├─ DRAM (HBM) 最高 → HBM 带宽瓶颈
  │   ├─ 检查 L2 Hit Rate: 低 = 频繁穿透到 DRAM
  │   ├─ 增加 L2 命中率(数据局部性/persisting cache)
  │   └─ 减少总 gmem 访问量(tiling)
  ├─ L2 最高 → L2 带宽/容量瓶颈
  │   ├─ 检查 L2 Compression (可压缩数据?)
  │   └─ 增大 tile 减少 L2 访问
  ├─ L1/TEX 最高 → L1/smem 瓶颈
  │   ├─ 检查 bank conflict
  │   ├─ 检查 sectors/request (coalescing)
  │   └<arg_value> 减少 smem 访问次数
  └─ Mem Pipes Busy 高 → 内存指令发射太多
      └─ 减少 memory op 数量(向量化/合并)
```

## 特殊: WGMMA (Hopper) 的注意事项

- WGMMA 是 **warpgroup 级**指令,4 个 scheduler 协同 → `warpgroup_arrive` stall 是特有的
- WGMMA 异步执行: 发出后 warp stall 等结果 → 表现为 `barrier` 或 `wait` stall
- WGMMA 直接读 smem(不经过 rmem) → `long_scoreboard` 可能指向 smem 读
- 低 occupancy + 高 SM Busy 是 WGMMA kernel 的**正常状态** → 不要盲目增加 occupancy
- WGMMA 累加器占大量寄存器 → 154 regs/thread 是典型的,减 tile 才能降 regs

---

## 16b. 指标后缀含义（理解原始指标名的关键）

用 `--query-metrics` 列出的 5623 个原始指标名都有后缀,理解后缀才能读懂:

| 后缀 | 含义 | 用途 |
|---|---|---|
| `.per_cycle_active` | 每个活跃周期的值 | 去掉了 SM idle 时间,反映真正工作时的速率 |
| `.per_cycle_elapsed` | 每个经过周期的值 | 包含 SM idle 时间,反映整体效率 |
| `.pct_of_peak_sustained_active` | 占活跃期峰值的% | 排除 idle,反映真实利用率 |
| `.pct_of_peak_sustained_elapsed` | 占经过期峰值的% | 包含 idle,反映整体利用率 |
| `.sum` | 总和 | 累积量 |
| `.avg` | 平均值 | 平均量 |
| `.max` | 最大值 | 跨 SM/scheduler 的最大值(看负载不均衡) |
| `.min` | 最小值 | 最小值 |

**Active vs Elapsed 的区别**:
- **Active**: 只算 SM 活跃的周期(SM Active Cycles)→ 反映「SM 干活时的效率」
- **Elapsed**: 算所有经过的周期(Elapsed Cycles)→ 反映「包含 idle 的整体效率」

例: `sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak_sustained_active` = tensor core 在 SM 活跃期的峰值利用率
= 90.97%(我们的值)→ SM 干活时,TC 几乎满载

如果看 `.pct_of_peak_sustained_elapsed` 会更低: 90.97% × 92.4%(active ratio) = 84.1% → 包含 idle 后的整体利用率。

**前缀含义**:
| 前缀 | 硬件层 |
|---|---|
| `sm__` | SM 级(SM 整体) |
| `smsp__` | SMSP 级(SM sub-partition,每 SM 4 个 SMSP) |
| `warp__` | Warp 级 |
| `l1tex__` | L1/TEX 缓存 |
| `lts__` | L2 缓存 |
| `dram__` | DRAM/HBM |
| `sm__pipe_*` | 计算管道(FP16/FP32/INT/TC) |

---

# Part V: 实战

## 17. ncu 工作原理 (Kernel Replay)

ncu 不能一次读完所有计数器——GPU 的 PMU 寄存器有限,一次只能读一部分。

```
第 1 遍: 启动 kernel → 读计数器组 A → 结束 → 恢复写入的内存
第 2 遍: 启动 kernel → 读计数器组 B → 结束 → 恢复写入的内存
第 3 遍: 启动 kernel → 读计数器组 C → 结束 → 恢复写入的内存
...
(~40 遍 for --set full on H20)
```

这就是为什么 `--set full` 慢——kernel 要跑很多遍。前提是 kernel 是
**deterministic 的**(同样的输入同样的输出),否则 replay 会出错。

### 三种 Replay 模式

| 模式 | 原理 | 适用场景 |
|---|---|---|
| **Kernel Replay** (默认) | 保存/恢复 kernel 写入的内存,同一 kernel 多次重跑 | 大多数 kernel |
| **Application Replay** | 整个程序重跑,每遍读一组 | kernel 有 host 交互/非确定性 |
| **Range Replay** | 捕获一段 API 调用范围重跑 | 需要并发 kernel / 不支持 kernel replay |

## 18. 实战工作流

### 步骤 1: Roofline 分析(理论)

先算清楚你的 kernel 在 compute/memory 哪条线上:

```
Arith intensity = FLOPs / gmem_bytes
  AI > 10  → compute-bound(优化计算)
  AI < 2   → memory-bound(优化访存)
```

### 步骤 2: ncu basic 快速看整体

```bash
$NCU --set basic -k regex:"gemm_kernel" -c 1 \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096
```

看 Speed of Light 确认 compute/memory 哪条线满。

### 步骤 3: 看 stall 原因

```bash
$NCU --section WarpStateStats --section SpeedOfLight \
  -k regex:"gemm_kernel" -c 1 \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096
```

找 top stall reason,理解 warp 为什么在等,对照 [Part III](#part-iii-warp-stall-reasons-全表)。

### 步骤 4: 提假设 → 改 kernel → 重测 → 对比

```
假设: "stall_barrier 占 74% → pipeline 同步开销"
  ↓
修改: 增大 NUM_STAGES / 改 producer 策略 / epilogue overlap
  ↓
验证: 重跑 ncu,对比 stall_barrier 从 74% 降到 50%
  ↓
确认: TFLOPS 提升了 → 假设正确
```

### 步骤 5: full set + GUI 深入

```bash
$NCU --set full -k regex:"gemm_kernel" -c 1 \
  --export /tmp/gemm_full.ncu-rep \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096
# 下载 .ncu-rep 到本地,用 Nsight Compute GUI 打开
# GUI 有 roofline plot、source attribution、memory chart 等
```

## 19. 常见坑

| 坑 | 原因 | 解决 |
|---|---|---|
| `ERR_NVGPUCTRPERM` | PMU 被禁(容器/驱动) | `--privileged` + driver 不限制 |
| `Failed to prepare kernel` | ncu 版本与 driver 不匹配 | 装 ncu 新版本 |
| 太慢 | `--set full` replay 40 遍 | 先 `--set basic` 或挑 `--section` |
| 输出太多 | 每次 launch 都 profile | 加 `--launch-count 1` + `--kernel-name` |
| profile 到 torch kernel | 没过滤 | 加 `--kernel-name regex:"gemm_kernel"` |
| kernel 非确定性 | replay 模式不适用 | `--replay-mode application`(重跑整个程序) |
| SM Freq 不一致 | benchmark 用 boost,ncu 用 sustained | 用 ncu 的 1.80 GHz 算真实 peak |
| Occupancy 低但性能高 | WGMMA 长指令 + pipeline | 低 occupancy 是 Hopper 正常状态,不盲目加 occupancy |

## 20. 速查卡片

### 命令速查

```bash
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu

# 一行搞定(full):
$NCU --set full --target-processes all -k regex:"gemm_kernel" -s 0 -c 1 \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096

# 快速版(basic):
$NCU --set basic -k regex:"gemm_kernel" -c 1 \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096

# 导出 GUI:
$NCU --set full -k regex:"gemm_kernel" -c 1 \
  --export /tmp/gemm.ncu-rep \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096

# 看 stall:
$NCU --section WarpStateStats -k regex:"gemm_kernel" -c 1 \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096

# 看 occupancy:
$NCU --section Occupancy -k regex:"gemm_kernel" -c 1 \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096

# 查所有可用指标:
$NCU --query-metrics 2>&1 | grep -i tensor | head -10

# 只看特定指标:
$NCU --metrics sm__pipe_tensor_op_hmma_cycles_active.active,\
smsp__pcsamp_warps_issue_stalled_barrier.sum \
  -k regex:"gemm_kernel" -c 1 \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096
```

### 关键指标速查

| 指标 | 含义 | 什么值要注意 |
|---|---|---|
| `Compute (SM) Throughput` | SM 计算吞吐占峰值% | >90% = 计算封顶;<50% = 未利用 |
| `Memory Throughput` | 整体内存吞吐% | >80% = 内存瓶颈;<50% = 不是瓶颈 |
| `DRAM Throughput` | HBM 带宽% | >80% = HBM 瓶颈 |
| `No Eligible` | 没 warp 可发射的周期% | >50% = 延迟隐藏不足(WGMMA 场景可能正常) |
| `Warp Cycles Per Issued Inst` | 指令间等待周期 | >30 = 需要大量 warp 并行 |
| `Theoretical Occupancy` | 理论占用率 | <50% 看什么限制了(regs/smem) |
| `Achieved Occupancy` | 实际占用率 | <<Theoretical = 负载不均 |
| `Registers Per Thread` | 寄存器用量 | >100 = occupancy 受限;增大会 spill |
| `SM Load Imbalance` | SM 间不均 | >10% = grid 尾数问题 |
| `L2 Hit Rate` | L2 命中率 | <20% = 频繁穿透到 DRAM |
| `Branch Efficiency` | 分支命中率 | <95% = 有 divergence |
| `stalled_barrier` | CTA barrier stall | 高 = pipeline/同步开销 |
| `stalled_long_scoreboard` | 内存依赖 stall | 高 = load 延迟未隐藏 |
| `stalled_imc_miss` | HBM miss stall | 高 = 数据不在 L2 |
| `stalled_not_selected` | 未被选中(正常) | 高 = 有足够并行 warp(好事) |

### stall 原因优先级

```
需要关注:     barrier > long_scoreboard > imc_miss > lg_throttle
正常(不需关注): not_selected > math_pipe_throttle > misc
特殊:         warpgroup_arrive (WGMMA) > drain > wait > no_instruction
```

---

## 21. 高级功能

### Profile Series（自动参数扫描）

ncu 支持 **Profile Series**——自动扫描多个配置,对比不同参数的性能:

```bash
$NCU --set basic --kernel-name regex:"gemm_kernel" \
  --profile-from 1 --profile-to 3 \
  .venv/bin/python ops/gemm/run_gemm.py 4096 4096 4096
```

| 参数 | 说明 |
|---|---|
| `--profile-from N` | 从第 N 次 kernel launch 开始 |
| `--profile-to N` | 到第 N 次 kernel launch 结束 |
| `--profile-series` | 启用 series 对比模式 |

GUI 中 Profile Series 显示为多列对比表,快速看参数变化对性能的影响。

### Clock Control（时钟锁定）

```bash
$NCU --clock-control none        # 不控制时钟(默认)
$NCU --clock-control boost       # 锁定到 boost clock
$NCU --clock-control base        # 锁定到 base clock
```

**为什么用**:不锁定时钟会导致 profiling 结果不可重现——GPU 频率随温度/负载波动。锁定后每次跑结果一致。

**代价**:锁定 boost 会让 GPU 发热,可能触发降频。锁定 base 低估性能。

### Cache Control（缓存控制）

```bash
$NCU --cache-control all         # 控制所有 cache
$NCU --cache-control none        # 不控制 cache
```

控制 cache 行为(如 L2 flush),确保每次 kernel replay 的 cache 状态一致。对 memory-bound kernel 的可重现性很重要。

### NVLink / NUMA Affinity

多 GPU 系统中,ncu 有 NVLink 和 NUMA 相关的 section:

| 指标 | 含义 |
|---|---|
| `NVLink Throughput` | NVLink 带宽利用率(多 GPU 通信) |
| `NVLink Data Transmitted` | NVLink 传输数据量 |
| `NUMA Affinity` | GPU 到 CPU 的 NUMA 距离(影响 P2P/统一内存性能) |

单 GPU GEMM 用不到这些,但多 GPU collective(MPI/NCCL)或多卡 split-K 时要看。

### Occupancy Calculator（GUI only）

GUI 有一个可视化的 **Occupancy Calculator**——拖动 slider 改 block size / registers / smem,实时看 occupancy 变化:

- 输入:block threads、registers/thread、dynamic smem
- 输出:theoretical occupancy、block limit、active warps
- 用途:探索「减少 2 个 regs 会怎样?」「smem 减半 occupancy 能涨多少?」

CLI 用 `--section Occupancy` 看 summary,但交互式 calculator 只在 GUI。

---

## 参考

- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA Compute Triage Guide](https://docs.nvidia.com/cuda/developer-preview/13.4/nsight-compute/ComputeTriage/index.html)
- [ncu Metrics Reference (社区)](https://the-dsvolk.github.io/ai-perf/profiling/ncu_metrics.html)
- [NVIDIA Developer Forums - Warp State Statistics](https://forums.developer.nvidia.com/t/how-are-the-cycles-of-different-warp-stall-reasons-calculated-in-the-section-warp-state-statistics/227170)
