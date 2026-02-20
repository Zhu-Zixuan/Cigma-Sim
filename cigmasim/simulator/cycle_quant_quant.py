# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Cycle counting for quantized-integer GEMM."""

import torch
from torch import Tensor

from cigmasim.config import (
    ArrayCountingConfig,
    ArraySchedulingMethod,
    CycleCountingConfig,
    PEProcessingStrategy,
)
from cigmasim.core import (
    count_array_cycles_vector_parallel,
    count_cycles_bit_parallel,
    count_cycles_bit_serial,
)
from cigmasim.type import QuantDataWrap, get_signed_digits


# @torch.compile(**compile_kwargs)
def cycle_simulator_quant_quant(
    a: QuantDataWrap,
    wt: QuantDataWrap,
    segment_len: int,
    cycle_config: CycleCountingConfig,
    array_config: ArrayCountingConfig,
) -> Tensor:
    """Simulate cycle count for quantized-integer GEMM.

    Args:
        a: Data wrap for `a` with shape `[..., M, K]`.
        wt: Data wrap for `wt` with shape `[..., N, K]`.
        segment_len: Length of each segment.
        cycle_config: PE cycle counting configuration.
        array_config: PE array configuration.

    Returns:
        Array-level cycle counts with shape `[...]`.

    """
    torch._check(a.shape[:-2] == wt.shape[:-2])
    torch._check(a.shape[-1] == wt.shape[-1])

    # --- 1. Resolve digit layout from encoding and format ---

    a_encode, w_encode = cycle_config.a_encode, cycle_config.w_encode

    # --- 2. Build segment view and expand PE pair dimensions ---

    # shape: [..., M, SEG, LEN] -> [..., M, 1, SEG, LEN]
    a = a.build_segment(segment_len).unsqueeze(-3)
    # shape: [..., N, SEG, LEN] -> [..., 1, N, SEG, LEN]
    wt = wt.build_segment(segment_len).unsqueeze(-4)

    # --- 3. Encode bit patterns into signed-digit nonzero planes ---

    # shape: [..., M, 1, SEG, LEN] -> [..., M, 1, SEG, LEN]
    a_nonzero = get_signed_digits(a.payload, enc=a_encode).nonzero
    # shape: [..., 1, N, SEG, LEN] -> [..., 1, N, SEG, LEN]
    wt_nonzero = get_signed_digits(wt.payload, enc=w_encode).nonzero

    # --- 4. Broadcast to MxN pairs and apply value skip ---

    target_shape = torch.broadcast_shapes(a_nonzero.shape, wt_nonzero.shape)

    # shape: [..., M, 1, SEG, LEN] -> [..., M, N, SEG, LEN]
    a_nonzero = a_nonzero.broadcast_to(target_shape)
    # shape: [..., 1, N, SEG, LEN] -> [..., M, N, SEG, LEN]
    wt_nonzero = wt_nonzero.broadcast_to(target_shape)

    if cycle_config.enable_value_skip:
        a_not_zero = a.payload != 0
        wt_nonzero = wt_nonzero * a_not_zero

    # --- 5. Count segment cycles in each PE ---

    # shape: [..., M, N, SEG, LEN] -> [..., M, N, SEG]
    match cycle_config.method:
        case PEProcessingStrategy.W_BIT_SERIAL:
            segment_cycles = count_cycles_bit_serial(
                data=wt_nonzero,
                digit_per_data=w_encode.get_digits(wt.fmt.bits),
                digit_width=w_encode.digit_width,
                lane_count=cycle_config.lane_count,
                window_size=cycle_config.window_size,
                mapping_order=cycle_config.mapping_order,
                lane_sharing=cycle_config.lane_sharing,
            )

        case PEProcessingStrategy.W_BIT_PARALLEL:
            segment_cycles = count_cycles_bit_parallel(
                data=wt_nonzero,
                digit_per_data=w_encode.get_digits(wt.fmt.bits),
                digit_width=w_encode.digit_width,
                lane_count=cycle_config.lane_count,
                window_size=cycle_config.window_size,
                mapping_order=cycle_config.mapping_order,
                lane_sharing=cycle_config.lane_sharing,
            )

        case _:
            raise NotImplementedError(f"Count cycle function not implemented for PE method: {cycle_config.method}")

    # --- 6. Aggregate PE cycles to array-level cycles ---

    # shape: [..., M, N, SEG] -> [...]
    match array_config.method:
        case ArraySchedulingMethod.VECTOR_PARALLEL_SEGMENT_SERIAL:
            array_cycles = count_array_cycles_vector_parallel(segment_cycles, array_config)
        case _:
            raise ValueError(f"Count array function not implemented for array method: {array_config.array_method}")

    return array_cycles
