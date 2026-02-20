# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Array-level cycle aggregation for PE arrays."""

import torch
import torch.nn as nn
from torch import Tensor

from cigmasim.config import ArrayCountingConfig


def count_array_cycles_vector_parallel(
    segment_cycles: Tensor,
    array_dims: ArrayCountingConfig,
) -> Tensor:
    """Aggregate segment-level cycles into final array cycles."""
    # --- 1. Reduce segment serial dimension ---

    # shape: [..., M, N, SEG] -> [..., M, N]
    cycles: Tensor = segment_cycles.sum(dim=-1, dtype=torch.int32)

    # --- 2. Pad and tile to physical array size ---

    a_size, w_size = array_dims.a_size, array_dims.w_size

    m_tiles = (cycles.size(-2) + a_size - 1) // a_size
    n_tiles = (cycles.size(-1) + w_size - 1) // w_size

    a_pad = m_tiles * a_size - cycles.size(-2)
    w_pad = n_tiles * w_size - cycles.size(-1)

    # shape: [..., M, N] -> [..., M_padded, N_padded]
    cycles = nn.functional.pad(cycles, (0, w_pad, 0, a_pad))

    # shape: [..., M_padded, N_padded] -> [..., M_tiles, a_size, N_padded]
    cycles = cycles.unflatten(dim=-2, sizes=(-1, a_size))
    # shape: [..., M_tiles, a_size, N_padded] -> [..., M_tiles, a_size, N_tiles, w_size]
    cycles = cycles.unflatten(dim=-1, sizes=(-1, w_size))

    # --- 3. Reduce PE-parallel dimensions inside each tile ---

    # shape: [..., M_tiles, a_size, N_tiles, w_size] -> [..., M_tiles, N_tiles]
    cycles = cycles.amax(dim=(-3, -1))

    # --- 4. Accumulate tile-serial dimensions ---

    # shape: [..., M_tiles, N_tiles] -> [...]
    cycles = cycles.sum(dim=(-2, -1), dtype=torch.int32)

    return cycles
