# AGENTS.md — guide for AI agents (and humans) working in this repo

## What this repo is

`cutedsl-ops` is a **practice playground** for writing GPU operators with
[NVIDIA's CuTe DSL](https://github.com/NVIDIA/cutlass) (the
`nvidia-cutlass-dsl` Python package). Each operator lives under `ops/<name>/`
as a kernel scaffold (`*_kernel.py`) plus a runnable correctness/bench harness
(`run_*.py`). The harnesses are complete; the **device kernel bodies are
deliberately left as guided TODOs** for the learner to implement.

Target operators: `gemm`, `flash_attn`, `megamoe`.

## Environment

- Python 3.12 (`.python-version`). The DSL requires `>=3.10`.
- Package manager: **uv**. `uv sync` installs everything except a CUDA `torch`
  build — install that yourself first (`make install`, edit the index URL to
  match your driver's CUDA version).
- A CUDA GPU of compute capability **>= sm_80** is required to actually run the
  kernels. GEMM works on Ampere; FlashAttention/MegaMoE targets Hopper (sm_90).

## Daily commands

```bash
make sync        # uv sync deps
make quality     # ruff check + format --check (the "test" gate, no GPU needed)
make style       # ruff --fix + format
make run-gemm    # python ops/gemm/run_gemm.py   (needs GPU + implemented kernel)
```

**Always run `make quality` after editing Python.** It needs no GPU and is the
only CI-equivalent gate in this repo.

## CuTe DSL conventions (follow these in every kernel)

Mirror the patterns from `cutlass-notes/0X-*/cutedsl_*.py`. Verified API
surface for the installed `nvidia-cutlass-dsl>=4.7.0`:

```python
import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream

@cute.kernel
def my_kernel(mA: cute.Tensor, ..., tiled_mma: cute.TiledMma, is_x: cutlass.Constexpr[bool]):
    tid, _, _ = cute.arch.thread_idx()        # returns a 3-tuple
    bx, by, bz = cute.arch.block_idx()        # returns a 3-tuple
    gA = cute.local_tile(mA, tiler=(M, K), coord=(0, 0))   # keeps static tile shape
    thr_mma = tiled_mma.get_slice(tid)
    tCgA = thr_mma.partition_A(gA)             # (MMA, MMA_M, MMA_K)
    tCrA = tiled_mma.make_fragment_A(tCgA)
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)
    cute.autovec_copy(tCgA, tCrA)             # gmem -> rmem vectorized copy
    cute.copy(tiled_copy, src, dst)           # tile-level copy (g2s / s2r / r2g)
    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)   # D <- A*B + C (D,C may alias)

@cute.jit
def my_op(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream: CUstream, ...):
    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float16, (16, 8, 8))   # Ampere
    # op = cute.nvgpu.warpgroup.MmaF16BF16Op(...)                                    # Hopper
    tiled_mma = cute.make_tiled_mma(op, atom_layout_mnk=(1, 1, 1))
    num_threads = tiled_mma.size
    my_kernel(mA, mB, mC, tiled_mma, ...).launch(grid=(...), block=(num_threads, 1, 1), stream=stream)
```

Runtime/compile pattern (TVM-FFI, so the compiled callable takes bare
`torch.Tensor` and runs on `torch.cuda.current_stream()`):

```python
from common.cute_runtime import make_cute_tensor, make_stream

a = torch.empty(M, K, device="cuda", dtype=torch.half)
proto = (make_cute_tensor(a), make_cute_tensor(b), make_cute_tensor(c), make_stream(), M, N, K)
compiled = cute.compile(my_op, *proto)  # specialize on dtype/layout + Constexprs
compiled(a, b, c)  # bare torch tensors at call time
```

Helpers `make_cute_tensor` / `make_stream` and the correctness/timing helpers
(`relative_error`, `compare_tensor`, `cuda_bench`) live in `common/`. Reuse
them in every `run_*.py` — do not re-derive.

Key APIs by file (verified locations in the installed package):
- `cute.gemm`, `cute.copy`, `cute.autovec_copy`, `cute.prefetch` — `cutlass.cute.algorithm`
- `cute.local_tile`, `cute.make_layout`, `cute.make_tensor` — `cutlass.cute.core`
- `cute.make_tiled_mma`, `cute.make_tiled_copy`, `cute.make_tiled_copy_tv` — `cutlass.cute.atom`
- `cute.arch.thread_idx`, `cute.arch.block_idx`, `cute.arch.cp_async_*` — `cutlass.cute.arch`
- warp MMAs `cute.nvgpu.warp.{MmaF16BF16Op,MmaTF32Op,MmaFP8Op}` — `cutlass.cute.nvgpu.warp.mma`
- warpgroup MMAs `cute.nvgpu.warpgroup.{MmaF16BF16Op,MmaF8Op,MmaI8Op}` — `cutlass.cute.nvgpu.warpgroup.mma`
- elementwise math — `cutlass.cute.math` (imported as `cute.math`)

Dtypes: `cutlass.Float16`, `cutlass.BFloat16`, `cutlass.Float32`, `cutlass.TFloat32`.
Constexprs: `cutlass.Constexpr[T]` in signatures, `cutlass.const_expr(x)` in bodies.

## When implementing a kernel TODO

1. Read the numbered recipe in the `# TODO(practice)` comments — it lists the
   exact DSL calls in order.
2. Keep the **host side** (`@cute.jit` entry + `run_*.py`) as-is unless you are
   adding a new tiled copy / atom; the harness is already correct.
3. After implementing, run `make run-<op>`; the harness prints `Max diff`,
   `Mean diff`, `RE%` and a `Success`/`Failed` verdict against a torch
   reference. `RE < tol` passes.
4. Do not commit secrets. `opencode.json` is kept secret-free on purpose;
   provider credentials go in your global `~/.config/opencode/opencode.json`.

## Reference material

- `../cutlass-notes/0X-*` — step-by-step lessons (01 minimal gemm → 14 warp
  specialization), each with a `cutedsl_*.py` DSL port. The closest reference
  for each op here.
- `../cutlass/python/CuTeDSL/cutlass/cute/` — the installed DSL source. Grep
  it to confirm signatures before writing.
