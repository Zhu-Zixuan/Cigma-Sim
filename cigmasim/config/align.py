# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Exponent alignment configuration for floating-point accumulation."""

from enum import StrEnum, auto
from typing import NamedTuple


class FloatAlignMethod(StrEnum):
    """Exponent alignment strategies.

    Attributes:
        PRODUCT: Align at the product level; no mantissa pre-shifting.
        MASK: Use a bit mask to split shift between activation and weight.
        SHIFT_WEIGHT: Shift exponent difference on weight up to a threshold.
        SHIFT_ACTIVATION: Shift exponent difference on activation up to a threshold.

    """

    PRODUCT = auto()
    MASK = auto()
    SHIFT_WEIGHT = auto()
    SHIFT_ACTIVATION = auto()


class ExponentAlignConfig(NamedTuple):
    """Configuration for exponent alignment.

    Attributes:
        method: The alignment strategy to use.
        shift_mask: Bit mask for MASK method (selects bits assigned to weight shift).
        max_shift: Maximum shift amount for SHIFT_WEIGHT / SHIFT_ACTIVATION.
            Negative value means unlimited.
        a_extend_bits: Extra mantissa bits appended to activation before shifting.
        w_extend_bits: Extra mantissa bits appended to weight before shifting.

    """

    method: FloatAlignMethod

    shift_mask: int = 0
    max_shift: int = 0
    a_extend_bits: int = 0
    w_extend_bits: int = 0
