# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Value simulation for quantized-integer GEMM."""

import torch
from torch import Tensor

from cigmasim.config import ElementWiseOp, InternalPrecisionConfig
from cigmasim.type import QuantDataWrap, QuantFormat, clog2, keep_fraction, keep_width


# @torch.compile(**compile_kwargs)
def value_simulator_quant_quant(
    a: QuantDataWrap,
    wt: QuantDataWrap,
    segment_len: int,
    value_config: InternalPrecisionConfig,
    trailing_op: ElementWiseOp = lambda x: x,
) -> Tensor:
    """Simulate quantized-integer GEMM output values.

    Args:
        a: Data wrap for `a` with shape `[..., M, K]`.
        wt: Data wrap for `wt` with shape `[..., N, K]`.
        segment_len: Length of each segment.
        value_config: Internal precision configuration.
        trailing_op: Post-multiply operation to apply.

    Returns:
        Simulated output values with shape `[..., M, N]`.

    """
    torch._check(a.shape[:-2] == wt.shape[:-2])
    torch._check(a.shape[-1] == wt.shape[-1])

    # --- 1. Build segment view and expand PE pair dimensions ---

    # shape: [..., M, SEG, LEN] -> [..., M, 1, SEG, LEN]
    a = a.build_segment(segment_len).unsqueeze(-3)
    # shape: [..., N, SEG, LEN] -> [..., 1, N, SEG, LEN]
    wt = wt.build_segment(segment_len).unsqueeze(-4)

    # --- 2. Compute element-wise products with internal precision ---

    fmt = QuantFormat(a.fmt.bits + wt.fmt.bits, a.fmt.frac + wt.fmt.frac)

    # shape: [..., M, N, SEG, LEN]
    product = a.payload.to(fmt.dtype) * wt.payload.to(fmt.dtype)
    product, fmt = keep_fraction(product, fmt, value_config.prod_frac, extend=False)

    # --- 3. Reduce inside each segment ---

    fmt = QuantFormat(fmt.bits + clog2(product.size(-1)), fmt.frac)

    # shape: [..., M, N, SEG]
    summation = product.to(fmt.dtype).sum(dim=-1, dtype=fmt.dtype)
    summation, fmt = keep_fraction(summation, fmt, value_config.summ_frac, extend=False)

    # --- 4. Accumulate across segments and requantize ---

    fmt = QuantFormat(fmt.bits + clog2(summation.shape[-1]), fmt.frac)

    # shape: [..., M, N]
    accumulation = summation.to(fmt.dtype).sum(dim=-1, dtype=fmt.dtype)
    accumulation, fmt = keep_fraction(accumulation, fmt, a.fmt.frac, mode=value_config.rounding_mode)

    result, fmt = keep_width(accumulation, fmt, a.fmt.bits)

    # --- 5. Apply trailing element-wise operation ---

    # shape: [..., M, N]
    result = trailing_op(result)

    return result
