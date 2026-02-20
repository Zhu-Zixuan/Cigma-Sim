# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Exponent alignment utilities for floating-point accumulation."""

import torch
from torch import Tensor

from cigmasim.config import ExponentAlignConfig, FloatAlignMethod


def compute_align_shifts(
    exp_sum: Tensor,
    config: ExponentAlignConfig,
) -> tuple[Tensor, Tensor]:
    """Compute activation/weight right-shift plans from exponent sums."""
    # --- 1. Compute relative exponent gap to running segment max ---
    exp_cummax = torch.amax(exp_sum, dim=-1, keepdim=True).cummax(dim=-2).values
    relative_exp = exp_cummax - exp_sum
    zeros = torch.zeros_like(exp_sum)

    # --- 2. Split relative shift by alignment policy ---
    match config.method:
        case FloatAlignMethod.MASK:
            alignment_mask = config.shift_mask
            w_shift = relative_exp & alignment_mask
            a_shift = relative_exp - w_shift
            return a_shift, w_shift

        case FloatAlignMethod.SHIFT_WEIGHT:
            if config.max_shift <= 0:
                return zeros, relative_exp

            w_shift = relative_exp.clamp_max(config.max_shift)
            a_shift = relative_exp - w_shift
            return a_shift, w_shift

        case FloatAlignMethod.SHIFT_ACTIVATION:
            if config.max_shift <= 0:
                return relative_exp, zeros

            a_shift = relative_exp.clamp_max(config.max_shift)
            w_shift = relative_exp - a_shift
            return a_shift, w_shift

        case _:
            raise NotImplementedError(f"Alignment method not implemented: {config.method}")
