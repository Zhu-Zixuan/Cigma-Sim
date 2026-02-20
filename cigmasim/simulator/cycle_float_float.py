# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Cycle counting for floating-point GEMM."""

import torch
from torch import Tensor

from cigmasim.config import (
    ArrayCountingConfig,
    ArraySchedulingMethod,
    CycleCountingConfig,
    ExponentAlignConfig,
    FloatAlignMethod,
    PEProcessingStrategy,
)
from cigmasim.core import (
    compute_align_shifts,
    count_array_cycles_vector_parallel,
    count_cycles_bit_parallel,
    count_cycles_bit_serial,
)
from cigmasim.type import (
    FloatDataWrap,
    get_compute_integer_dtype,
    get_signed_digits,
    logical_right_shift,
)


# @torch.compile(**compile_kwargs)
def cycle_simulator_float_float(
    a: FloatDataWrap,
    wt: FloatDataWrap,
    segment_len: int,
    align_config: ExponentAlignConfig,
    cycle_config: CycleCountingConfig,
    array_config: ArrayCountingConfig,
) -> Tensor:
    """Simulate cycle count for floating-point GEMM.

    Args:
        a: Data wrap for `a` with shape `[..., M, K]`.
        wt: Data wrap for `wt` with shape `[..., N, K]`.
        segment_len: Length of each segment.
        align_config: Exponent alignment configuration.
        cycle_config: PE cycle counting configuration.
        array_config: PE array configuration.

    Returns:
        Array-level cycle counts with shape `[...]`.

    """
    torch._check(a.shape[:-2] == wt.shape[:-2])
    torch._check(a.shape[-1] == wt.shape[-1])

    # --- 1. Resolve mantissa digit layout from encoding and format ---

    a_encode, w_encode = cycle_config.a_encode, cycle_config.w_encode
    a_extend_bits, w_extend_bits = align_config.a_extend_bits, align_config.w_extend_bits

    a_man_width = 2 + a.fmt.mantissa_bits + a_extend_bits
    w_man_width = 2 + wt.fmt.mantissa_bits + w_extend_bits

    # --- 2. Build segment view and expand PE pair dimensions ---

    # shape: [..., M, SEG, LEN] -> [..., M, 1, SEG, LEN]
    a = a.build_segment(segment_len).unsqueeze(-3)
    # shape: [..., N, SEG, LEN] -> [..., 1, N, SEG, LEN]
    wt = wt.build_segment(segment_len).unsqueeze(-4)

    # --- 3. Extract mantissa nonzero planes and product exponents ---

    # sign does not affect nonzero digit pattern

    # shape: [..., M, 1, SEG, LEN] -> [..., M, 1, SEG, LEN]
    a_nonzero = get_signed_digits(a.full_mantissa, enc=a_encode).nonzero
    a_exponent = a.adjusted_exponent
    # shape: [..., 1, N, SEG, LEN] -> [..., 1, N, SEG, LEN]
    wt_nonzero = get_signed_digits(wt.full_mantissa, enc=w_encode).nonzero
    wt_exponent = wt.adjusted_exponent

    # --- 4. Broadcast to MxN pairs and apply value skip ---

    target_shape = torch.broadcast_shapes(a_nonzero.shape, wt_nonzero.shape)

    # shape: [..., M, 1, SEG, LEN] -> [..., M, N, SEG, LEN]
    a_nonzero = a_nonzero.broadcast_to(target_shape)
    # shape: [..., 1, N, SEG, LEN] -> [..., M, N, SEG, LEN]
    wt_nonzero = wt_nonzero.broadcast_to(target_shape)
    # shape: [..., M, N, SEG, LEN]
    p_exponent = a_exponent + wt_exponent

    if cycle_config.enable_value_skip:
        a_not_zero = a.unsigned_field != 0
        wt_nonzero = wt_nonzero * a_not_zero

    # --- 5. Align mantissas by exponent policy ---

    if align_config.method != FloatAlignMethod.PRODUCT:
        a_man_dtype = get_compute_integer_dtype(a_man_width)
        w_man_dtype = get_compute_integer_dtype(w_man_width)

        a_shift, w_shift = compute_align_shifts(p_exponent, align_config)

        # must be logical right shift, 1 can locate at MSB
        a_nonzero = logical_right_shift(a_nonzero.to(a_man_dtype) << a_extend_bits, a_shift, a_man_width)
        wt_nonzero = logical_right_shift(wt_nonzero.to(w_man_dtype) << w_extend_bits, w_shift, w_man_width)

    # --- 6. Count segment cycles in each PE ---

    # shape: [..., M, N, SEG, LEN] -> [..., M, N, SEG]
    match cycle_config.method:
        case PEProcessingStrategy.W_BIT_SERIAL:
            segment_cycles = count_cycles_bit_serial(
                data=wt_nonzero,
                digit_per_data=w_encode.get_digits(w_man_width),
                digit_width=w_encode.digit_width,
                lane_count=cycle_config.lane_count,
                window_size=cycle_config.window_size,
                mapping_order=cycle_config.mapping_order,
                lane_sharing=cycle_config.lane_sharing,
            )

        case PEProcessingStrategy.W_BIT_PARALLEL:
            segment_cycles = count_cycles_bit_parallel(
                data=wt_nonzero,
                digit_per_data=w_encode.get_digits(w_man_width),
                digit_width=w_encode.digit_width,
                lane_count=cycle_config.lane_count,
                window_size=cycle_config.window_size,
                mapping_order=cycle_config.mapping_order,
                lane_sharing=cycle_config.lane_sharing,
            )

        case _:
            raise NotImplementedError(f"Count cycle function not implemented for PE method: {cycle_config.method}")

    # --- 7. Aggregate PE cycles to array-level cycles ---

    # shape: [..., M, N, SEG] -> [...]
    match array_config.method:
        case ArraySchedulingMethod.VECTOR_PARALLEL_SEGMENT_SERIAL:
            array_cycles = count_array_cycles_vector_parallel(segment_cycles, array_config)
        case _:
            raise ValueError(f"Count array function not implemented for array method: {array_config.array_method}")

    return array_cycles
