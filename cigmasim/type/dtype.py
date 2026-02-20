# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Helpers for mapping bit width to torch dtypes."""

import torch


def get_storage_integer_dtype(bit_width: int) -> torch.dtype:
    """Return the smallest signed dtype that stores `bit_width` bits."""
    if 0 < bit_width <= 8:
        return torch.int8
    if bit_width <= 16:
        return torch.int16
    if bit_width <= 32:
        return torch.int32
    if bit_width <= 64:
        return torch.int64
    raise ValueError(f"Unsupported bit width: {bit_width}")


def get_compute_integer_dtype(bit_width: int) -> torch.dtype:
    """Return the smallest compute dtype for `bit_width` bits."""
    if 0 < bit_width <= 32:
        return torch.int32
    if bit_width <= 64:
        return torch.int64
    raise ValueError(f"Unsupported bit width: {bit_width}")
