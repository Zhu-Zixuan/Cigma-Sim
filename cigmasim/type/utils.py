# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Tensor utils."""

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


def logical_right_shift(t: Tensor, shifts: Tensor, width: int) -> Tensor:
    """Logical right shift for signed integer tensors."""
    # s <= 0:    mask = 11111111 : ~(0)
    # 0 < s < N: mask = 00011111 : ~(-1 << (N-s))
    # N <= s:    mask = 00000000 : ~(-1 << 0)
    mask = (-1) << torch.clamp(width - shifts, 0, width - 1)
    mask = ~torch.where(shifts <= 0, 0, mask)

    t = t >> shifts.clamp(0, width - 1)
    t = t & mask
    return t


def arithmetic_right_shift(t: Tensor, shifts: Tensor, width: int) -> Tensor:
    """Arithmetic right shift for signed integer tensors."""
    t = t >> shifts.clamp(0, width - 1)
    t = torch.where(shifts >= width, 0, t)
    return t
