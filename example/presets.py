# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Architecture preset factories for experiments."""

from collections.abc import Callable
from functools import partial

from torch import Tensor

from cigmasim import FLOAT32, ArrayCountingConfig, ArraySchedulingMethod, FloatFormat, QuantFormat

from .architectures import (
    bitlet_cycle,
    bitlet_value,
    cigma_cycle,
    cigma_value,
)

# --- type aliases ---

CycleFn = Callable[[Tensor, Tensor], Tensor]

ValueFn = Callable[[Tensor, Tensor], Tensor]

# --- defaults ---

DEFAULT_ARRAY_CONFIG = ArrayCountingConfig(
    method=ArraySchedulingMethod.VECTOR_PARALLEL_SEGMENT_SERIAL,
    a_size=1,
    w_size=1,
)

INT16Q12 = QuantFormat(bits=16, frac=12)

# --- registries ---

QUANT_ARCH_NAMES: list[str] = [
    "cigma",
    "bitlet",
    "tetris",
    "dwp",
    "dwp_intra",
    "pragmatic",
    "tactical",
]

FLOAT_ARCH_NAMES: list[str] = ["cigma", "bitlet"]

_QUANT_CYCLE_REGISTRY: dict[str, Callable] = {
    "cigma": cigma_cycle,
    "bitlet": bitlet_cycle,
}

_FLOAT_CYCLE_REGISTRY: dict[str, Callable] = {
    "cigma": cigma_cycle,
    "bitlet": bitlet_cycle,
}

_QUANT_VALUE_REGISTRY: dict[str, Callable] = {
    "cigma": cigma_value,
    "bitlet": bitlet_value,
}

_FLOAT_VALUE_REGISTRY: dict[str, Callable] = {
    "cigma": cigma_value,
    "bitlet": bitlet_value,
}


# --- getters ---


def get_quant_cycle_fn(
    arch: str,
    array_config: ArrayCountingConfig = DEFAULT_ARRAY_CONFIG,
    quant_fmt: QuantFormat = INT16Q12,
) -> CycleFn:
    """Return quantized cycle function for one architecture."""
    fn = _QUANT_CYCLE_REGISTRY[arch]
    return partial(fn, a_fmt=quant_fmt, w_fmt=quant_fmt, array_config=array_config)


def get_quant_value_fn(
    arch: str,
    quant_fmt: QuantFormat = INT16Q12,
) -> ValueFn | None:
    """Return quantized value function for one architecture."""
    fn = _QUANT_VALUE_REGISTRY.get(arch)
    if fn is None:
        return None
    return partial(fn, a_fmt=quant_fmt, w_fmt=quant_fmt)


def get_float_cycle_fn(
    arch: str,
    array_config: ArrayCountingConfig = DEFAULT_ARRAY_CONFIG,
    float_fmt: FloatFormat = FLOAT32,
) -> CycleFn:
    """Return floating-point cycle function for one architecture."""
    fn = _FLOAT_CYCLE_REGISTRY[arch]
    return partial(fn, a_fmt=float_fmt, w_fmt=float_fmt, array_config=array_config)


def get_float_value_fn(
    arch: str,
    float_fmt: FloatFormat = FLOAT32,
) -> ValueFn:
    """Return floating-point value function for one architecture."""
    fn = _FLOAT_VALUE_REGISTRY[arch]
    return partial(fn, a_fmt=float_fmt, w_fmt=float_fmt)
