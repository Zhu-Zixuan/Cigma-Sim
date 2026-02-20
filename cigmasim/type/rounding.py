# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Rounding utils."""

from enum import StrEnum, auto

import torch
from torch import Tensor


class RoundingMode(StrEnum):
    """Rounding modes.

    Attributes:
        FULL_DOWN: Round towards -inf (Floor).
        FULL_CEIL: Round towards +inf (Ceil).
        FULL_TO_ZERO: Round towards zero (Truncate).
        FULL_TO_INF: Round away from zero.
        HALF_DOWN: Round to nearest, ties to -inf.
        HALF_CEIL: Round to nearest, ties to +inf.
        HALF_TO_ZERO: Round to nearest, ties to zero.
        HALF_TO_INF: Round to nearest, ties away from zero.
        HALF_TO_EVEN: Round to nearest, ties to even.
        HALF_TO_ODD: Round to nearest, ties to odd.

    """

    FULL_DOWN = auto()
    FULL_CEIL = auto()
    FULL_TO_ZERO = auto()
    FULL_TO_INF = auto()
    HALF_DOWN = auto()
    HALF_CEIL = auto()
    HALF_TO_ZERO = auto()
    HALF_TO_INF = auto()
    HALF_TO_EVEN = auto()
    HALF_TO_ODD = auto()


def round_offset_complement(
    complement: Tensor,
    drop_shift: int,
    mode: RoundingMode,
) -> Tensor:
    """Compute rounding offset for two's-complement fixed-point inputs."""
    # --- 1. Build bit masks and boolean predicates ---

    lsb_mask = 1 << drop_shift  # LSB of integer part (?x.????)
    drop_mask = lsb_mask - 1  # All drop part (??.xxxx)
    guard_mask = lsb_mask >> 1  # Guard bit, (??.x???)

    sign = complement < 0
    lsb_is_odd = (complement & lsb_mask) != 0
    has_drop = (complement & drop_mask) != 0
    has_guard = (complement & guard_mask) != 0
    has_round = has_drop & ~has_guard

    # truth table for FULL_*
    #
    # | value | integer | S | F | down | ceil | zero | inf |
    # | >+a.0 |  +a     | 0 | 1 |  +0  |  +1  |  +0  | +1  |
    # |  +a.0 |  +a     | 0 | 0 |  +0  |  +0  |  +0  | +0  |
    # |  -a.0 |  -a     | 1 | 0 |  +0  |  +0  |  +0  | +0  |
    # | <-a.0 |  -a-1   | 1 | 1 |  +0  |  +1  |  +1  | +0  |

    # truth table for HALF_*
    #
    # | value | integer | S | G | R | down | ceil | zero | inf |  even  |   odd  |
    # | >+a.5 |  +a     | 0 | 1 | 1 |  +1  |  +1  |  +1  | +1  |   +1   |   +1   |
    # |  +a.5 |  +a     | 0 | 1 | 0 |  +0  |  +1  |  +0  | +1  | +lsb/e | +lsb/o |
    # | <+a.5 |  +a     | 0 | 0 | 1 |  +0  |  +0  |  +0  | +0  |   +0   |   +0   |
    # |  +a.0 |  +a     | 0 | 0 | 0 |  +0  |  +0  |  +0  | +0  |   +0   |   +0   |
    # |  -a.0 |  -a     | 1 | 0 | 0 |  +0  |  +0  |  +0  | +0  |   +0   |   +0   |
    # | >-a.5 |  -a-1   | 1 | 1 | 1 |  +1  |  +1  |  +1  | +1  |   +1   |   +1   |
    # |  -a.5 |  -a-1   | 1 | 1 | 0 |  +0  |  +1  |  +1  | +0  | +lsb/e | +lsb/o |
    # | <-a.5 |  -a-1   | 1 | 0 | 1 |  +0  |  +0  |  +0  | +0  |   +0   |   +0   |
    #
    # | value | integer | lsb/e | lsb/o | S | G | R | even | odd |
    # |  +a.5 |  +a     |   1   |   0   | 0 | 1 | 0 |  +1  | +0  |
    # |  +a.5 |  +a     |   0   |   1   | 0 | 1 | 0 |  +0  | +1  |
    # |  -a.5 |  -a-1   |   0   |   1   | 1 | 1 | 0 |  +0  | +1  |
    # |  -a.5 |  -a-1   |   1   |   0   | 1 | 1 | 0 |  +1  | +0  |

    # --- 2. Resolve mode-specific rounding offset ---

    offset: Tensor

    match mode:
        case RoundingMode.FULL_DOWN:  # round towards -inf
            offset = torch.zeros_like(complement)

        case RoundingMode.FULL_CEIL:  # round towards +inf
            offset = has_drop

        case RoundingMode.FULL_TO_ZERO:  # round towards zero
            offset = has_drop & sign

        case RoundingMode.FULL_TO_INF:  # round away from zero
            offset = has_drop & ~sign

        case RoundingMode.HALF_DOWN:  # round to nearest, ties to -inf
            offset = has_guard & has_round

        case RoundingMode.HALF_CEIL:  # round to nearest, ties to +inf
            offset = has_guard

        case RoundingMode.HALF_TO_ZERO:  # round to nearest, ties to zero
            offset = has_guard & (has_round | sign)

        case RoundingMode.HALF_TO_INF:  # round to nearest, ties away from zero
            offset = has_guard & (has_round | ~sign)

        case RoundingMode.HALF_TO_EVEN:  # round to nearest, ties to even
            offset = has_guard & (has_round | lsb_is_odd)

        case RoundingMode.HALF_TO_ODD:  # round to nearest, ties to odd
            offset = has_guard & (has_round | ~lsb_is_odd)

        case _:
            raise ValueError(f"Unsupported rounding mode: {mode}")

    return offset


def round_offset_true_form(
    magnitude: Tensor,
    sign_bool: Tensor,
    drop_shift: Tensor,
    mode: RoundingMode,
) -> Tensor:
    """Compute rounding offset for sign-magnitude fixed-point inputs."""
    # --- 1. Build bit masks and boolean predicates ---

    lsb_mask = 1 << drop_shift  # LSB of integer part (?x.????)
    drop_mask = lsb_mask - 1  # All drop part (??.xxxx)
    guard_mask = lsb_mask >> 1  # Guard bit, (??.x???)

    lsb_is_odd = (magnitude & lsb_mask) != 0
    has_drop = (magnitude & drop_mask) != 0
    has_guard = (magnitude & guard_mask) != 0
    has_round = has_drop & ~has_guard

    # truth table for FULL_*
    #
    # | value | integer | S | F | down | ceil | zero | inf |
    # | >+a.0 |  +a     | 0 | 1 |  +0  |  +1  |  +0  | +1  |
    # |  +a.0 |  +a     | 0 | 0 |  +0  |  +0  |  +0  | +0  |
    # |  -a.0 |  -a     | 1 | 0 |  -0  |  -0  |  -0  | -0  |
    # | <-a.0 |  -a     | 1 | 1 |  -1  |  -0  |  -0  | -1  |

    # truth table for HALF_*
    #
    # | value | integer | S | G | R | down | ceil | zero | inf |  even  |   odd  |
    # | >+a.5 |  +a     | 0 | 1 | 1 |  +1  |  +1  |  +1  | +1  |   +1   |   +1   |
    # |  +a.5 |  +a     | 0 | 1 | 0 |  +0  |  +1  |  +0  | +1  | +lsb/e | +lsb/o |
    # | <+a.5 |  +a     | 0 | 0 | 1 |  +0  |  +0  |  +0  | +0  |   +0   |   +0   |
    # |  +a.0 |  +a     | 0 | 0 | 0 |  +0  |  +0  |  +0  | +0  |   +0   |   +0   |
    # |  -a.0 |  -a     | 1 | 0 | 0 |  -0  |  -0  |  -0  | -0  |   -0   |   -0   |
    # | >-a.5 |  -a     | 1 | 0 | 1 |  -0  |  -0  |  -0  | -0  |   -0   |   -0   |
    # |  -a.5 |  -a     | 1 | 1 | 0 |  -1  |  -0  |  -0  | -1  | -lsb/e | -lsb/o |
    # | <-a.5 |  -a     | 1 | 1 | 1 |  -1  |  -1  |  -1  | -1  |   -1   |   -1   |
    #
    # | value | integer | lsb/e | lsb/o | S | G | R | even | odd |
    # |  +a.5 |  +a     |   1   |   0   | 0 | 1 | 0 |  +1  | +0  |
    # |  +a.5 |  +a     |   0   |   1   | 0 | 1 | 0 |  +0  | +1  |
    # |  -a.5 |  -a     |   0   |   1   | 1 | 1 | 0 |  -0  | -1  |
    # |  -a.5 |  -a     |   1   |   0   | 1 | 1 | 0 |  -1  | -0  |

    # --- 2. Resolve mode-specific rounding offset ---

    offset: Tensor

    match mode:
        case RoundingMode.FULL_DOWN:  # round towards -inf
            offset = has_drop & sign_bool

        case RoundingMode.FULL_CEIL:  # round towards +inf
            offset = has_drop & ~sign_bool

        case RoundingMode.FULL_TO_ZERO:  # round towards zero
            offset = torch.zeros_like(magnitude, dtype=torch.bool)

        case RoundingMode.FULL_TO_INF:  # round away from zero
            offset = has_drop

        case RoundingMode.HALF_DOWN:  # round to nearest, ties to -inf
            offset = has_guard & (has_round | sign_bool)

        case RoundingMode.HALF_CEIL:  # round to nearest, ties to +inf
            offset = has_guard & (has_round | ~sign_bool)

        case RoundingMode.HALF_TO_ZERO:  # round to nearest, ties to zero
            offset = has_guard & has_round

        case RoundingMode.HALF_TO_INF:  # round to nearest, ties away from zero
            offset = has_guard

        case RoundingMode.HALF_TO_EVEN:  # round to nearest, ties to even
            offset = has_guard & (has_round | lsb_is_odd)

        case RoundingMode.HALF_TO_ODD:  # round to nearest, ties to odd
            offset = has_guard & (has_round | ~lsb_is_odd)

        case _:
            raise ValueError(f"Unsupported rounding mode: {mode}")

    return offset
