# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Configuration types for alignment, cycles, arrays, and precision."""

from .align import ExponentAlignConfig, FloatAlignMethod
from .array import ArrayCountingConfig, ArraySchedulingMethod
from .cycle import CycleCountingConfig, PEMappingOrder, PEProcessingStrategy
from .value import InternalPrecisionConfig

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
)
