# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Core functions for cycle and value computation."""

from .align import compute_align_shifts
from .count_array import count_array_cycles_vector_parallel
from .count_cycle import count_cycles_bit_parallel, count_cycles_bit_serial

__all__ = [
    # align
    "compute_align_shifts",
    # count array
    "count_array_cycles_vector_parallel",
    # count cycle
    "count_cycles_bit_parallel",
    "count_cycles_bit_serial",
]
