"""FlashAttention kernel scaffolds (exercises 1-5).

Each module exposes a host-side ``@cute.jit`` entry point and a device
``@cute.kernel`` whose body is a guided TODO.  See ``reference.py`` for the
torch reference implementations used by the harnesses.

Exercises (ordered by difficulty):
  1. ``prefill_bf16_multistage`` — FA v2 prefill, single-WG, kStage=2  (hpc-ops A1)
  2. ``prefill_bf16_warpspec``   — warp-specialized prefill             (hpc-ops A2)
  3. ``decode_bf16_splitk``      — decode, split-K + paged KV + combine (hpc-ops D1)
  4. ``prefill_fp8``             — FP8 paged prefill, per-tensor scales (hpc-ops C1)
  5. ``decode_fp8``              — FP8 decode, QK=SS SV=RS             (hpc-ops D2)
"""
