# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Configuration types for alignment, cycles, arrays, and precision."""

from collections.abc import Callable

from torch import Tensor

from .align import ExponentAlignConfig, FloatAlignMethod
from .array import ArrayCountingConfig, ArraySchedulingMethod
from .cycle import CycleCountingConfig, PEMappingOrder, PEProcessingStrategy
from .value import InternalPrecisionConfig

ElementWiseOp = Callable[[Tensor], Tensor]

__all__ = (
    # align
    "FloatAlignMethod",
    "ExponentAlignConfig",
    # array
    "ArraySchedulingMethod",
    "ArrayCountingConfig",
    # cycle
    "PEProcessingStrategy",
    "PEMappingOrder",
    "CycleCountingConfig",
    # value
    "InternalPrecisionConfig",
    # element-wise op
    "ElementWiseOp",
)
