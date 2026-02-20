# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Data types, formats, and encoding helpers for the simulator."""

from .dtype import get_compute_integer_dtype
from .encoding import (
    DigitType,
    Encoding,
    QuinaryDigits,
    TernaryDigits,
    get_signed_digits,
)
from .float_type import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    FP8_E5M2,
    FloatDataWrap,
    FloatFormat,
)
from .quant_type import (
    QuantDataWrap,
    QuantFormat,
    check_overflow,
    clog2,
    keep_fraction,
    keep_width,
)
from .rounding import RoundingMode
from .utils import arithmetic_right_shift, logical_right_shift

__all__ = [
    # dtype
    "get_compute_integer_dtype",
    # encoding
    "DigitType",
    "Encoding",
    "QuinaryDigits",
    "TernaryDigits",
    "get_signed_digits",
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
    # rounding
    "RoundingMode",
    # utils
    "arithmetic_right_shift",
    "logical_right_shift",
    "clog2",
    "keep_fraction",
    "keep_width",
    "check_overflow",
]
