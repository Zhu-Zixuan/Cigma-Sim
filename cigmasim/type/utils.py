# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Tensor utils for matrix simulation."""

import torch
from torch import Tensor


def build_segment(t: Tensor, length: int) -> Tensor:
    """Split the last dimension into fixed-size segments."""
    if length <= 0:
        raise ValueError(f"segment_len ({length}) <= 0")

    count = (t.size(-1) + length - 1) // length
    padding = count * length - t.size(-1)

    t = torch.nn.functional.pad(t, (0, padding))
    t = t.unflatten(dim=-1, sizes=(-1, length))

    return t


def logical_right_shift(t: Tensor, shifts: Tensor) -> Tensor:
    """Logical right shift for signed integer tensors."""
    bit_width = t.element_size() * 8
    full_one = torch.tensor(-1, dtype=t.dtype, device=t.device)
    mask = full_one << torch.clamp(bit_width - shifts, 0, bit_width - 1)
    mask = ~(mask.where(shifts <= 0, 0))
    t = (t >> shifts.clamp(0, bit_width - 1)) & mask
    return t
