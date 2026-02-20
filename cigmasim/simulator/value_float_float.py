# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Value simulation for floating-point GEMM."""

import torch

from cigmasim.config import ExponentAlignConfig, FloatAlignMethod, InternalPrecisionConfig
from cigmasim.core import compute_align_shifts
from cigmasim.type import FloatDataWrap, QuantFormat, arithmetic_right_shift, clog2, keep_fraction


# @torch.compile(**compile_kwargs)
def value_simulator_float_float(
    a: FloatDataWrap,
    wt: FloatDataWrap,
    segment_len: int,
    align_config: ExponentAlignConfig,
    value_config: InternalPrecisionConfig,
) -> FloatDataWrap:
    """Simulate floating-point GEMM values with architectural effects.

    Args:
        a: Data wrap for `a` with shape `[..., M, K]`.
        wt: Data wrap for `wt` with shape `[..., N, K]`.
        segment_len: Length of each segment.
        align_config: Exponent alignment configuration.
        value_config: Internal precision configuration.

    Returns:
        Simulated output values with shape `[..., M, N]`.

    """
    torch._check(a.shape[:-2] == wt.shape[:-2])
    torch._check(a.shape[-1] == wt.shape[-1])

    # --- 1. Derive internal fixed-point formats ---

    a_extend_bits, w_extend_bits = align_config.a_extend_bits, align_config.w_extend_bits
    a_mantissa_bits, w_mantissa_bits = a.fmt.mantissa_bits + a_extend_bits, wt.fmt.mantissa_bits + w_extend_bits

    prod_fmt = QuantFormat(3 + a_mantissa_bits + w_mantissa_bits, a_mantissa_bits + w_mantissa_bits)

    # --- 2. Build segment view and expand PE pair dimensions ---

    # shape: [..., M, SEG, LEN] -> [..., M, 1, SEG, LEN]
    a = a.build_segment(segment_len).unsqueeze(-3)
    # shape: [..., N, SEG, LEN] -> [..., 1, N, SEG, LEN]
    wt = wt.build_segment(segment_len).unsqueeze(-4)

    # --- 3. Extract mantissas/exponents/signs and broadcast ---

    # shape: [..., M, 1, SEG, LEN] -> [..., M, 1, SEG, LEN]
    a_mantissa = a.full_mantissa.to(prod_fmt.dtype)
    a_exponent = a.unbiased_exponent
    # shape: [..., 1, N, SEG, LEN] -> [..., 1, N, SEG, LEN]
    wt_mantissa = wt.full_mantissa.to(prod_fmt.dtype)
    wt_exponent = wt.unbiased_exponent

    target_shape = torch.broadcast_shapes(a_mantissa.shape, wt_mantissa.shape)

    # shape: [..., M, N, SEG, LEN]
    a_mantissa = a_mantissa.broadcast_to(target_shape)
    wt_mantissa = wt_mantissa.broadcast_to(target_shape)
    prod_exponent = a_exponent + wt_exponent
    prod_sign = a.sign_bool ^ wt.sign_bool

    # --- 4. Align mantissas by architecture policy ---

    # shape: [..., M, N, SEG, LEN]
    if align_config.method != FloatAlignMethod.PRODUCT:
        a_man_width = 2 + a_mantissa_bits
        w_man_width = 2 + w_mantissa_bits

        a_shift, w_shift = compute_align_shifts(prod_exponent, align_config)

        # use arithmetic right shift, mantissa is signed value
        a_mantissa = arithmetic_right_shift(a_mantissa << a_extend_bits, a_shift, a_man_width)
        wt_mantissa = arithmetic_right_shift(wt_mantissa << w_extend_bits, w_shift, w_man_width)

    # --- 5. Compute products and apply product precision ---

    # shape: [..., M, N, SEG, LEN]
    product = wt_mantissa * a_mantissa
    if value_config.prod_frac is not None:
        product, prod_fmt = keep_fraction(product, prod_fmt, value_config.prod_frac)

    prod_sign_value = 1 - 2 * prod_sign.to(product.dtype)
    product = product * prod_sign_value

    # --- 6. Track progressive segment exponent baseline ---

    # shape: [..., M, N, SEG]
    summ_exponent = prod_exponent.amax(dim=-1).cummax(dim=-1).values

    # --- 7. Align products to segment baseline (PRODUCT mode only) ---

    if align_config.method == FloatAlignMethod.PRODUCT:
        prod_shift = summ_exponent.unsqueeze(-1) - prod_exponent
        product = arithmetic_right_shift(product, prod_shift, prod_fmt.bits)

    # --- 8. Sum within each segment and quantize ---

    summ_fmt = QuantFormat(prod_fmt.bits + clog2(product.size(-1)), prod_fmt.frac)

    # shape: [..., M, N, SEG]
    summation = product.sum(dim=-1, dtype=summ_fmt.dtype)
    if value_config.summ_frac is not None:
        summation, summ_fmt = keep_fraction(summation, summ_fmt, value_config.summ_frac)

    # --- 9. Accumulate segments with progressive alignment ---

    # shape: [..., M, N, SEG-1]
    exp_step = summ_exponent.diff(dim=-1)

    # shape: [..., M, N]
    accu_fmt = summ_fmt
    accumulation = summation[..., 0]
    accu_exponent = summ_exponent[..., -1]

    # Drop bits segment-by-segment to emulate hardware accumulation.
    # Under `torch.compile`, this loop is typically unrolled.
    for i in range(1, summation.shape[-1]):
        accumulation = arithmetic_right_shift(accumulation, exp_step[..., i - 1], accu_fmt.bits)
        accumulation = accumulation + summation[..., i]

    # --- 10. Convert accumulated fixed-point result to float ---

    # shape: [..., M, N]
    result = FloatDataWrap.from_parts(
        accumulation < 0, accu_exponent - accu_fmt.frac, accumulation.abs(), a.fmt, rounding=value_config.rounding_mode
    )

    return result
