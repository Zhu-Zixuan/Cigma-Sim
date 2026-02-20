# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""PE array configuration for cycle aggregation."""

from enum import StrEnum, auto
from typing import NamedTuple


class ArraySchedulingMethod(StrEnum):
    """Scheduling methods for PE array execution.

    Attributes:
        VECTOR_PARALLEL_SEGMENT_SERIAL: Vectors are processed in parallel
            across PEs; segments of a vector are processed serially.

    """

    VECTOR_PARALLEL_SEGMENT_SERIAL = auto()


class ArrayCountingConfig(NamedTuple):
    """Configuration for array-level cycle counting.

    Attributes:
        method: The array scheduling method.
        a_size: Number of PEs along the activation (row) dimension.
        w_size: Number of PEs along the weight (column) dimension.

    """

    method: ArraySchedulingMethod

    a_size: int
    w_size: int
