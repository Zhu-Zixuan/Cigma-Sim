# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Internal precision configuration for value simulation."""

from typing import NamedTuple

from cigmasim.type import RoundingMode


class InternalPrecisionConfig(NamedTuple):
    """Configuration for internal computation precision.

    Attributes:
        rounding_mode: Rounding mode for fixed-point truncation.
        product_frac: Target fractional bits for element-wise products.
            Negative value means no truncation.
        summation_frac: Target fractional bits for intra-segment sums.
            Negative value means no truncation.

    """

    rounding_mode: RoundingMode

    prod_frac: int | None = None
    summ_frac: int | None = None
