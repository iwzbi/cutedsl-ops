# FlashAttention — CuTe DSL 练习

FA v2 前向 pass 的 CuTe DSL 实现练习集，对标 [hpc-ops](https://github.com/Tencent/hpc-ops)
的 attention 内核家族，从零实现 Hopper `sm_90` CUDA varlen prefill 内核
（WGMMA + 在线 softmax）。

## 文件结构

```
ops/flash_attn/
├── README.md                     # 本文件
├── reference.py                  # torch 参考实现 + 对比工具
│                                 #   ref_decode_bf16 / ref_decode_fp8
│                                 #   pack_varlen       (flatten+pad64 打包)
│                                 #   allclose          (fp32 对比 + top-N 报错)
│                                 #   lse_combine       (decode split-K 合并)
│                                 #   repeat_kv         (GQA 广播)
├── compare_hpcops.py             # 与 hpc-ops 的精度 + 性能对比（varlen causal）
├── run_prefill.py                # prefill 练习 (ex.1) 正确性 + bench harness
├── run_decode.py                 # decode 练习 (ex.3/5) 正确性 + bench harness
├── kernels/                      # 每个练习一个 kernel 模块（class-based）
│   ├── prefill_bf16_multistage.py  # ex.1 varlen multi-stage prefill（已验证 PASS）
│   ├── decode_bf16_splitk.py       # ex.3  decode split-K + paged KV（scaffold）
│   └── decode_fp8.py               # ex.5  FP8 decode（scaffold）
└── tests/
    └── test_varlen.py            # varlen kernel 冒烟测试
```

每个 kernel 都是**类**：`@cute.jit __call__`（host：MMMA 组装、SMEM 布局、
SharedStorage 分配、launch）+ `@cute.kernel kernel`（device 主体）。调用约定：

```python
inst = FlashAttnPrefillBf16Multistage()
compiled = cute.compile(inst, mQ, mK, mV, mO, mSeqlens, mCuSeqlens, stream, ...)  # compile(实例, ...)
compiled(q, k, v, o, seqlens, cu_seqlens)  # 调用裸 torch.Tensor
```

## 练习映射（对标 hpc-ops attention 家族）

| ex | 本仓库 kernel            | hpc-ops 内核              | 语义                                |
|----|--------------------------|---------------------------|-------------------------------------|
| 1  | `prefill_bf16_multistage`| A1 `multi_stage_dim128`   | **varlen** prefill, 单 WG, 多 stage |
| 3  | `decode_bf16_splitk`     | D1 `smallm_bf16_dim128_static` | decode, split-K, paged KV + LSE combine |
| 5  | `decode_fp8`             | D2 `smallm_fp8_..._static`| FP8 decode, QK=SS + SV=RS           |

练习形态统一为 **varlen**：prefill 只保留 ex.1 varlen multi-stage 内核（原
dense 变体与 dense 形态 scaffold 均已删除，见 git 历史）。

刻意不做：decode 动态调度（E1-E3，`task_map`/PDL 是 host 工程而非 kernel 学习点）、
block-sparse（C3/C4，难度过高，留给 FP8 之后）。

## API 约定

**varlen prefill**（`ex.1`，全仓库 prefill 主形态）：镜像 hpc-ops
`attention_prefill_bf16` —— 输入 `(total_seq, H_q, D)` Q / `(total_seq, H_kv, D)`
K·V + `seqlens` + `cu_seqlens`；grid `(B*H_q, ceil(max_seqlens/64), 1)`；
**always causal**（无 `is_causal` 参数）；内部 scale `1/sqrt(D)`；V 按 batch pad
为 `(B, H_kv, D, max_seqlens)`。

要求 **每个 batch 的 flattened 起始偏移（即 `cu_seqlens`）为 64 的倍数**——
kernel 以 64 行为粒度切分 gmem 且 mask 用 batch-local 坐标；若 batch 边界未
对齐，tile 会跨 batch、mask 与实际 gmem 行错位（静默错误结果）。因此 harness
在 flatten 时通过 `pack_varlen` 把每个 batch 的 Q/K/O 段补零到 64 对齐，
`cu_seqlens` 传补齐后的偏移（`seqlens` 仍为真实长度）。

- **decode** (`ex.3/5`)：paged KV `(num_pages, H_kv, page_size, D)` + `block_table`
  `(B, max_blocks)` int32，grid `(kSplitK, BH, 1)`，输出 `(kSplitK, BH, blk_m, D)`
  部分结果 + `(kSplitK, BH, blk_m)` LSE，host 侧 `lse_combine` 合并。

## 精度验证工作流

```bash
make quality                              # ruff 检查/格式（无需 GPU，唯一 CI 门）
python ops/flash_attn/run_prefill.py                    # ex.1 varlen 正确性(全部 PREFILL_SHAPES)
python ops/flash_attn/tests/test_varlen.py               # varlen 正确性(5 case, 含非对齐 seq)
python ops/flash_attn/compare_hpcops.py [--shapes 0,1]   # 3-way: torch SDPA / hpc-ops / 本仓库
```

参考实现 (`reference.py`)：bf16 用 torch SDPA（`is_causal`、`scale` 可选），
GQA 自动 `repeat_kv` 广播；FP8 参考**刻意复刻 kernel 数值流程**（P×256 量化后再乘 V），
而非纯 fp32 softmax。`allclose(ref, out, atol=...)` 在 fp32 下比较，失败打印 top-10 误差。

## 性能工作流

每个 kernel 一个 `PERFLOG_<kernel>.md`（对标 `ops/gemm/PERFLOG.md` 风格），记录：
当前有 [`PERFLOG_prefill_multistage.md`](./PERFLOG_prefill_multistage.md)（ex.1 varlen multi-stage）；
decode 等 kernel 开工时再各开一份。

1. 硬件基线（H20: sm_90, 78 SM, 148 TFLOPS FP16 峰值）
2. Master Performance Table（shape × 版本列，TFLOPS + 相对峰值 %）
3. 与 hpc-ops 的同输入对比（varlen causal 对齐）
4. 每个优化步骤一个 git tag（`flash-exx-vN-<opt>`）+ 优化原理
5. `ncu_reports/` 性能剖析证据

```bash
python ops/flash_attn/run_prefill.py --bench    # TFLOPS bench（同一份 PREFILL_SHAPES）
python ops/flash_attn/run_prefill.py --ncu      # ncu 剖析
```