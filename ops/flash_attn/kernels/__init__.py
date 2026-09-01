"""FlashAttention kernels — varlen prefill (exercise 1) + decode scaffolds.

Each module exposes a host-side ``@cute.jit`` entry point and a device
``@cute.kernel`` whose body is a guided TODO.  See ``reference.py`` for the
torch reference implementations used by the harnesses.

The project is varlen-first: prefill (ex.1) is a single multi-stage varlen
kernel.  Remaining dense-form scaffolds were dropped (git history has them).

Exercises:
  1. ``prefill_bf16_multistage`` — FA v2 varlen prefill, single-WG, kStage=2 (hpc-ops A1)
  3. ``decode_bf16_splitk``      — decode, split-K + paged KV + combine (hpc-ops D1)
  5. ``decode_fp8``              — FP8 decode, QK=SS SV=RS             (hpc-ops D2)
"""
