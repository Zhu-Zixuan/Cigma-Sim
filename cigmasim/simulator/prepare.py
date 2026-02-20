# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Prepare tensors for simulator functions."""

from typing import TypedDict

import torch._dynamo
from torch import Tensor


class _CompileOptions(TypedDict, total=False):
    fullgraph: bool
    dynamic: bool
    mode: str
    backend: str
    options: dict[str, str | int | bool]


compile_kwargs: _CompileOptions = {
    "fullgraph": True,
    "dynamic": False,
    "mode": "max-autotune",
    # "mode": "reduce-overhead",
}


def validate_and_prepare(a: Tensor, wt: Tensor) -> tuple[Tensor, Tensor]:
    """Validate and prepare input tensors for simulation.

    Args:
        a: `a` tensor with shape [..., M, K].
        wt: `wt` tensor with shape [..., N, K].

    Returns:
        Prepared `(a, wt)` tensors with flattened batch dimensions and dynamic dimension marks.

    Raises:
        ValueError: If shapes are incompatible.

    """
    # --- Check tensor shape ---

    if a.ndim != wt.ndim or a.ndim < 2:
        raise ValueError(f"a.ndim ({a.ndim}) not match w.ndim ({wt.ndim})")

    if a.shape[:-2] != wt.shape[:-2] or a.shape[-1] != wt.shape[-1]:
        raise ValueError(f"a.shape ({a.shape}) not match wt.shape ({wt.shape})")

    # --- Flatten to ensure `ndim == 3` to benefit cuda graph ---

    a = a.reshape(-1, a.shape[-2], a.shape[-1])
    wt = wt.reshape(-1, wt.shape[-2], wt.shape[-1])

    # --- Mark dynamic dimensions to benefit cuda graph ---

    torch._dynamo.mark_dynamic(a, list(range(a.ndim)))
    torch._dynamo.mark_dynamic(wt, list(range(wt.ndim)))

    return a, wt
