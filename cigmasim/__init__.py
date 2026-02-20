# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""CigmaSim package for bit-sparsity aware GEMM simulation."""

from .config import (
    ArrayCountingConfig,
    ArraySchedulingMethod,
    CycleCountingConfig,
    ExponentAlignConfig,
    FloatAlignMethod,
    InternalPrecisionConfig,
    PEMappingOrder,
    PEProcessingStrategy,
)
from .simulator import (
    compile_kwargs,
    cycle_simulator_float_float,
    cycle_simulator_quant_quant,
    validate_and_prepare,
    value_simulator_float_float,
    value_simulator_quant_quant,
)
from .type import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    FP8_E5M2,
    Encoding,
    FloatDataWrap,
    FloatFormat,
    QuantDataWrap,
    QuantFormat,
    RoundingMode,
)

__all__ = [
    # align config
    "FloatAlignMethod",
    "ExponentAlignConfig",
    # array config
    "ArraySchedulingMethod",
    "ArrayCountingConfig",
    # cycle config
    "PEProcessingStrategy",
    "PEMappingOrder",
    "CycleCountingConfig",
    # value config
    "InternalPrecisionConfig",
    # encoding
    "Encoding",
    # rounding
    "RoundingMode",
    # data format
    "FloatFormat",
    "QuantFormat",
    "FLOAT64",
    "FLOAT32",
    "BFLOAT16",
    "FLOAT16",
    "FP8_E5M2",
    # data wrap
    "FloatDataWrap",
    "QuantDataWrap",
    # simulator
    "cycle_simulator_float_float",
    "cycle_simulator_quant_quant",
    "value_simulator_float_float",
    "value_simulator_quant_quant",
    # prepare
    "compile_kwargs",
    "validate_and_prepare",
]
