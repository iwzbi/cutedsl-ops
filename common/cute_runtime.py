"""CuTe DSL runtime helpers.

These wrappers mirror the TVM-FFI pattern from the cutlass-notes DSL ports so
that every pre-compiled callable accepts bare ``torch.Tensor`` arguments and
runs on ``torch.cuda.current_stream()`` without per-call DLPack plumbing or
explicit stream management.

Verified against ``nvidia-cutlass-dsl>=4.7.0``.
"""

from __future__ import annotations

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack, make_fake_stream


# Map a few torch dtype string names to CuTe element types. Useful when building
# MMA atoms from a torch dtype.
TORCH_TO_CUTE_DTYPE = {
    "float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
}


def make_cute_tensor(
    t,
    *,
    assumed_align: int = 16,
    leading_dim: int | None = None,
) -> cute.Tensor:
    """Wrap a torch tensor as a CuTe DSL tensor.

    ``leading_dim`` marks that axis as the contiguous (static-stride) axis so
    the DSL can infer strides at compile time without baking the full static
    shape in. Pass ``None`` to leave the layout fully dynamic.
    """
    ct = from_dlpack(t, assumed_align=assumed_align, enable_tvm_ffi=True)
    if leading_dim is not None:
        ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


def make_stream():
    """A TVM-FFI environment stream bound to ``torch.cuda.current_stream()``.

    Pass this to the host ``@cute.jit`` entry as the ``stream: CUstream`` arg.
    """
    return make_fake_stream(use_tvm_ffi_env_stream=True)


__all__ = ["TORCH_TO_CUTE_DTYPE", "make_cute_tensor", "make_stream"]
