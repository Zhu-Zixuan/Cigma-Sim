# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""CIGMA architecture configuration and simulation APIs."""

from dataclasses import dataclass

import torch
from torch import Tensor

from cigmasim import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    ArrayCountingConfig,
    CycleCountingConfig,
    Encoding,
    ExponentAlignConfig,
    FloatAlignMethod,
    FloatDataWrap,
    FloatFormat,
    InternalPrecisionConfig,
    PEMappingOrder,
    PEProcessingStrategy,
    QuantDataWrap,
    QuantFormat,
    RoundingMode,
    compile_kwargs,
    cycle_simulator_float_float,
    cycle_simulator_quant_quant,
    validate_and_prepare,
    value_simulator_float_float,
)

# --- architecture specific parameters ---


_a_encode = Encoding.TWOS_COMPLEMENT
_w_encode = Encoding.CSD_COMPACT

_enable_value_skip = True
_cycle_method = PEProcessingStrategy.W_BIT_SERIAL
_mapping_order = PEMappingOrder.STRIP

_align_config = ExponentAlignConfig(method=FloatAlignMethod.PRODUCT)

_float_rounding_mode = RoundingMode.FULL_TO_ZERO


# --- architecture specific API ---


@torch.compile(**compile_kwargs)
def cigma_cycle_quant_quant(
    a: Tensor,
    a_fmt: QuantFormat,
    wt: Tensor,
    w_fmt: QuantFormat,
    *,
    segment_len: int,
    lane_count: int,
    window_size: int,
    array_config: ArrayCountingConfig,
) -> Tensor:
    """Estimate CIGMA cycle count for quantized GEMM."""
    cycle_config = CycleCountingConfig(
        a_encode=_a_encode,
        w_encode=_w_encode,
        enable_value_skip=_enable_value_skip,
        method=_cycle_method,
        mapping_order=_mapping_order,
        lane_count=lane_count,
        window_size=window_size,
    )

    return cycle_simulator_quant_quant(
        a=QuantDataWrap.from_value(a, a_fmt),
        wt=QuantDataWrap.from_value(wt, w_fmt),
        segment_len=segment_len,
        cycle_config=cycle_config,
        array_config=array_config,
    )


@torch.compile(**compile_kwargs)
def cigma_cycle_float_float(
    a: Tensor,
    a_fmt: FloatFormat,
    wt: Tensor,
    w_fmt: FloatFormat,
    *,
    segment_len: int,
    lane_count: int,
    window_size: int,
    array_config: ArrayCountingConfig,
) -> Tensor:
    """Estimate CIGMA cycle count for floating-point GEMM."""
    cycle_config = CycleCountingConfig(
        a_encode=_a_encode,
        w_encode=_w_encode,
        enable_value_skip=_enable_value_skip,
        method=_cycle_method,
        mapping_order=_mapping_order,
        lane_count=lane_count,
        window_size=window_size,
    )

    return cycle_simulator_float_float(
        a=FloatDataWrap.from_value(a, a_fmt),
        wt=FloatDataWrap.from_value(wt, w_fmt),
        segment_len=segment_len,
        align_config=_align_config,
        cycle_config=cycle_config,
        array_config=array_config,
    )


@torch.compile(**compile_kwargs)
def cigma_value_float_float(
    a: Tensor,
    a_fmt: FloatFormat,
    wt: Tensor,
    w_fmt: FloatFormat,
    *,
    segment_len: int,
    product_frac: int,
) -> Tensor:
    """Simulate CIGMA floating-point GEMM output."""
    value_config = InternalPrecisionConfig(
        rounding_mode=_float_rounding_mode,
        prod_frac=product_frac,
    )

    result = value_simulator_float_float(
        a=FloatDataWrap.from_value(a, a_fmt),
        wt=FloatDataWrap.from_value(wt, w_fmt),
        segment_len=segment_len,
        align_config=_align_config,
        value_config=value_config,
    )

    return result.to_value()


# --- data specific parameters ---


@dataclass(frozen=True)
class _DataConfig:
    """Data-type-specific hardware parameters."""

    segment_len: int
    window_mult: int
    lane_count: int
    product_frac: int = 0


_data_configs: dict[tuple[FloatFormat | int, FloatFormat | int], _DataConfig] = {
    (8, 8): _DataConfig(segment_len=64, window_mult=8, lane_count=16),
    (16, 16): _DataConfig(segment_len=64, window_mult=4, lane_count=16),
    (BFLOAT16, BFLOAT16): _DataConfig(segment_len=64, window_mult=4, lane_count=16, product_frac=7 + 7 + 18),
    (FLOAT16, FLOAT16): _DataConfig(segment_len=64, window_mult=4, lane_count=16, product_frac=10 + 10 + 13),
    (FLOAT32, FLOAT32): _DataConfig(segment_len=64, window_mult=2, lane_count=32, product_frac=23 + 23 + 7),
}


# --- top API ---


def cigma_cycle(
    a: Tensor,
    wt: Tensor,
    *,
    a_fmt: FloatFormat | QuantFormat,
    w_fmt: FloatFormat | QuantFormat,
    array_config: ArrayCountingConfig,
) -> Tensor:
    """Dispatch CIGMA cycle simulation by `a_fmt`/`w_fmt`."""
    a, wt = validate_and_prepare(a, wt)

    if isinstance(a_fmt, QuantFormat) and isinstance(w_fmt, QuantFormat):
        cfg = _data_configs[(a_fmt.bits, w_fmt.bits)]

        return cigma_cycle_quant_quant(
            a,
            a_fmt,
            wt,
            w_fmt,
            segment_len=cfg.segment_len,
            lane_count=cfg.lane_count,
            window_size=_w_encode.get_digits(w_fmt.bits) * cfg.window_mult,
            array_config=array_config,
        )

    if isinstance(a_fmt, FloatFormat) and isinstance(w_fmt, FloatFormat):
        cfg = _data_configs[(a_fmt, w_fmt)]

        return cigma_cycle_float_float(
            a,
            a_fmt,
            wt,
            w_fmt,
            segment_len=cfg.segment_len,
            lane_count=cfg.lane_count,
            window_size=_w_encode.get_digits(2 + w_fmt.mantissa_bits) * cfg.window_mult,
            array_config=array_config,
        )

    raise TypeError(f"Incompatible format combination: ({a_fmt}, {w_fmt})")


def cigma_value(
    a: Tensor,
    wt: Tensor,
    *,
    a_fmt: FloatFormat,
    w_fmt: FloatFormat,
) -> Tensor:
    """Dispatch CIGMA value simulation by `a_fmt`/`w_fmt`."""
    a, wt = validate_and_prepare(a, wt)

    if isinstance(a_fmt, FloatFormat) and isinstance(w_fmt, FloatFormat):
        cfg = _data_configs[(a_fmt, w_fmt)]

        return cigma_value_float_float(
            a,
            a_fmt,
            wt,
            w_fmt,
            segment_len=cfg.segment_len,
            product_frac=cfg.product_frac,
        )

    raise TypeError(f"Incompatible format combination: ({a_fmt}, {w_fmt})")
