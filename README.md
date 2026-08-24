# cutedsl-ops

A practice playground for writing GPU operators with NVIDIA's
[CuTe DSL](https://github.com/NVIDIA/cutlass) — the `nvidia-cutlass-dsl`
Python package. Each operator ships a kernel **scaffold** plus a complete
torch-referenced correctness & timing harness; the device kernel body is left
as a guided TODO for you to implement and validate.

## Operators

| Op | Path | Reference idea | Min SM |
| --- | --- | --- | --- |
| GEMM | [`ops/gemm`](ops/gemm) | Block-tiled `C = A @ B^T` via `TiledMma` + `TiledCopy` | sm_80 |
| FlashAttention | [`ops/flash_attn`](ops/flash_attn) | Tiled online-softmax attention (`S = QK^T`, `O = PV`) | sm_90 |
| MegaMoE | [`ops/megamoe`](ops/megamoe) | Grouped GEMM + softmax gating (MoE FFN) | sm_90 |

Each `ops/<name>/` contains:
- `<name>_kernel.py` — `@cute.kernel` device kernel + `@cute.jit` host entry,
  with a numbered `# TODO(practice)` recipe listing the exact DSL calls.
- `run_<name>.py` — torch reference, `cute.compile(...)` specialization, launch,
  correctness check and (optional) timing.

## Setup

```bash
git clone <this repo> cutedsl-ops && cd cutedsl-ops

# 1. A CUDA-enabled torch matching your driver (edit the index URL as needed):
make install      # uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Everything else:
make sync         # uv sync

# 3. Check tooling (no GPU needed):
make quality
```

Requirements:
- Python 3.12 (`>=3.10` works).
- CUDA GPU, compute capability `>= sm_80` (GEMM) / `>= sm_90` (FlashAttn, MegaMoE).
- CUDA driver matching a `torch` CUDA build.

## How to practice

1. Open `ops/<name>/<name>_kernel.py` and follow the `# TODO(practice)` steps.
2. Run `make run-<name>`. The harness compiles your kernel once, launches it on
   a problem instance, and compares against a torch reference, printing
   `Max diff`, `Mean diff`, `RE%` and a `Success`/`Failed` verdict.
3. Iterate until `RE < tol`. Then tune: tile sizes, atom layout, pipelining,
   swizzling, warp specialization.

## Shared helpers

`common/` holds the reusable harness so every operator stays consistent:

- `common.cute_runtime` — `make_cute_tensor` (DLPack → CuTe tensor, TVM-FFI),
  `make_stream` (env stream bound to `torch.cuda.current_stream()`).
- `common.bench` — `relative_error`, `compare_tensor`, `cuda_bench`.

These mirror the TVM-FFI pattern from the
[cutlass-notes](https://github.com/ArthurinRUC/cutlass-notes) DSL ports so the
pre-compiled callables accept bare `torch.Tensor` args directly.

## Tooling

- **uv** for env/dep management (`uv.lock`).
- **ruff** for lint + format (`make style` / `make quality`).
- `make` targets drive the common workflow; see `make help`.

## License

MIT. Operator scaffolds reference NVIDIA's CuTe DSL under its
[EULA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html);
the DSL package itself is governed by that EULA, not this repo's license.
