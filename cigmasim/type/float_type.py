# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Floating-point format and bit-pattern wrapper definitions."""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from .dtype import get_compute_integer_dtype, get_storage_integer_dtype
from .rounding import RoundingMode, round_offset_true_form
from .utils import build_segment


@dataclass(frozen=True)
class FloatFormat:
    """Floating-point format definition.

    Attributes:
        exponent_bits: Number of bits in the exponent field.
        mantissa_bits: Number of bits in the mantissa field (excluding hidden bit).
        value_dtype: Corresponding PyTorch dtype for values.

    """

    exponent_bits: int
    mantissa_bits: int
    value_dtype: torch.dtype

    @property
    def total_bits(self) -> int:
        """Get total number of bits."""
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def unsigned_bits(self) -> int:
        """Get number of unsigned bits."""
        return self.exponent_bits + self.mantissa_bits

    @property
    def sign_mask(self) -> int:
        """Get sign field mask."""
        # IEEE 754: |s|_____|__________|
        return -(1 << self.unsigned_bits)

    @property
    def exponent_mask(self) -> int:
        """Get exponent field mask."""
        # IEEE 754: |_|eeeee|__________|
        return ((1 << self.exponent_bits) - 1) << self.mantissa_bits

    @property
    def mantissa_mask(self) -> int:
        """Get mantissa field mask."""
        # IEEE 754: |_|_____|tttttttttt|
        return (1 << self.mantissa_bits) - 1

    @property
    def unsigned_mask(self) -> int:
        """Get unsigned field mask."""
        # IEEE 754: |_|eeeee|tttttttttt|
        return (1 << self.unsigned_bits) - 1

    @property
    def hidden_bit(self) -> int:
        """Get implicit leading bit (hidden bit)."""
        # IEEE 754: |_|____1|__________|
        return 1 << self.mantissa_bits

    @property
    def exponent_bias(self) -> int:
        """Get exponent bias."""
        # IEEE 754: 2^(k-1)-1, |01111|
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def max_biased_exp(self) -> int:
        """Get biased exponent of maximum value."""
        # IEEE 754: 2^(k)-2, |11110|
        return self.exponent_bias * 2

    @property
    def max_unsigned(self) -> int:
        """Get unsigned field of maximum value."""
        # IEEE 754: |_|11110|1111111111|
        # NV E4M3: |_|1111|110| (dont support)
        return (self.max_biased_exp << self.mantissa_bits) | self.mantissa_mask

    @property
    def inf_unsigned(self) -> int:
        """Get unsigned field of inf."""
        # IEEE 754: |_|11111|0000000000|
        # NV E4M3: no inf (dont support)
        return self.exponent_mask

    @property
    def nan_unsigned(self) -> int:
        """Get unsigned field of nan."""
        # IEEE 754: |_|11111|1111111111|
        return self.exponent_mask | self.mantissa_mask

    @property
    def zero_unsigned(self) -> int:
        """Get unsigned field of zero."""
        # IEEE 754: |_|00000|0000000000|
        return 0

    # --- Dtype helpers ---

    @property
    def dtype(self) -> torch.dtype:
        """Get compute dtype for full binary bit pattern."""
        return get_compute_integer_dtype(self.total_bits)

    @property
    def storage_dtype(self) -> torch.dtype:
        """Get storage integer dtype for this format."""
        return get_storage_integer_dtype(self.total_bits)


# IEEE 754 binary64
FLOAT64 = FloatFormat(exponent_bits=11, mantissa_bits=52, value_dtype=torch.float64)
# IEEE 754 binary32
FLOAT32 = FloatFormat(exponent_bits=8, mantissa_bits=23, value_dtype=torch.float32)
# Google Brain Float16
BFLOAT16 = FloatFormat(exponent_bits=8, mantissa_bits=7, value_dtype=torch.bfloat16)
# IEEE 754 binary16
FLOAT16 = FloatFormat(exponent_bits=5, mantissa_bits=10, value_dtype=torch.float16)
# NVIDIA FP8 with 5-bit exponent
FP8_E5M2 = FloatFormat(exponent_bits=5, mantissa_bits=2, value_dtype=torch.float8_e5m2)
# NVIDIA FP8 with 4-bit exponent, not IEEE 754 compatible
# FP8_E4M3 = FloatFormat(exponent_bits=4, mantissa_bits=3, value_dtype=torch.float8_e4m3fn)

_DTYPE_TO_FLOAT_FORMAT = {
    torch.float64: FLOAT64,
    torch.float32: FLOAT32,
    torch.bfloat16: BFLOAT16,
    torch.float16: FLOAT16,
    torch.float8_e5m2: FP8_E5M2,
    # torch.float8_e4m3fn: FP8_E4M3,
}


@dataclass
class FloatDataWrap:
    """Wrapper for raw floating-point binary bit patterns.

    Attributes:
        payload: Integer tensor holding raw binary bit patterns.
        fmt: Floating-point format specification.

    """

    payload: Tensor
    fmt: FloatFormat

    # --- Tensor-like methods ---

    @property
    def shape(self) -> torch.Size:
        """Forward to `self.payload`."""
        return self.payload.shape

    @property
    def dtype(self) -> torch.dtype:
        """Forward to `self.payload`."""
        return self.payload.dtype

    @property
    def device(self) -> torch.device:
        """Forward to `self.payload`."""
        return self.payload.device

    def unsqueeze(self, dim: int) -> Self:
        """Forward to `self.payload` and re-wrap."""
        payload = self.payload.unsqueeze(dim)
        return self.__class__(payload, self.fmt)

    def broadcast_to(self, shape: torch.Size) -> Self:
        """Forward to `self.payload` and re-wrap."""
        payload = self.payload.broadcast_to(shape)
        return self.__class__(payload, self.fmt)

    def select(self, dim: int, index: int) -> Self:
        """Forward to `self.payload` and re-wrap."""
        payload = self.payload.select(dim, index)
        return self.__class__(payload, self.fmt)

    def clone(self) -> Self:
        """Forward to `self.payload` and re-wrap."""
        payload = self.payload.clone()
        return self.__class__(payload, self.fmt)

    def split(self, split_size: int, dim: int) -> tuple[Self, ...]:
        """Forward to `self.payload` and re-wrap."""
        chunks = self.payload.split(split_size, dim)
        return tuple(self.__class__(chunk, self.fmt) for chunk in chunks)

    # --- Build segment ---

    def build_segment(self, segment_len: int) -> Self:
        """Split the last dimension into fixed-size segments."""
        payload = build_segment(self.payload, segment_len)
        return self.__class__(payload, self.fmt)

    # --- Format conversion ---

    @classmethod
    def from_parts(
        cls,
        sign_bool: Tensor,
        exponent: Tensor,
        mantissa: Tensor,
        fmt: FloatFormat,
        *,
        rounding: RoundingMode = RoundingMode.HALF_TO_EVEN,
        inf_mask: Tensor | None = None,
        nan_mask: Tensor | None = None,
    ) -> Self:
        """Construct floating-point binary bit pattern from parts.

        The represented value is (-1)^sign * 2^exponent * mantissa.

        Args:
            sign_bool: Boolean tensor, False=positive, True=negative.
            exponent: Integer tensor, unbiased exponent.
            mantissa: Integer tensor, unsigned mantissa (integer form).
            fmt: Target floating-point format.
            rounding: Rounding mode for mantissa truncation.
            inf_mask: Inf mask.
            nan_mask: NaN mask.

        Returns:
            FloatDataWrap with the target format.

        """
        binary_field = parts_to_float_binary(
            sign_bool, exponent, mantissa, fmt, rounding=rounding, inf_mask=inf_mask, nan_mask=nan_mask
        )

        return cls(binary_field.to(fmt.storage_dtype), fmt)

    @classmethod
    def from_other(
        cls,
        other: Self,
        fmt: FloatFormat,
        rounding: RoundingMode = RoundingMode.HALF_TO_EVEN,
    ) -> Self:
        """Construct floating-point binary bit pattern from another format.

        Args:
            other: Source floating-point binary bit pattern.
            fmt: Target floating-point format.
            rounding: Rounding mode for mantissa truncation.

        Returns:
            FloatDataWrap with the target format.

        """
        if other.fmt == fmt:
            return other

        binary_field = parts_to_float_binary(
            other.sign_bool,
            other.unbiased_exponent.to(fmt.dtype) - other.fmt.mantissa_bits,
            other.full_mantissa.to(fmt.dtype),
            fmt,
            rounding=rounding,
            inf_mask=other.is_inf,
            nan_mask=other.is_nan,
        )

        return cls(binary_field.to(fmt.storage_dtype), fmt)

    @classmethod
    def from_value(cls, value: Tensor, fmt: FloatFormat) -> Self:
        """Construct floating-point binary bit pattern from value.

        Args:
            value: source floating-point value
            fmt: target floating-point format

        Returns:
            target floating-point binary bit pattern

        """
        if value.dtype not in _DTYPE_TO_FLOAT_FORMAT:
            raise ValueError(
                f"Unsupported dtype: {value.dtype}, the value Tensor must be a floating-point type. Use `from_other` for integer types."
            )
        value_fmt = _DTYPE_TO_FLOAT_FORMAT[value.dtype]
        value_binary = value.contiguous().view(value_fmt.storage_dtype)
        value_wrap = cls(value_binary, value_fmt)
        return cls.from_other(value_wrap, fmt)

    def to_value(self) -> Tensor:
        """Decode floating-point values from bit patterns."""
        scale = self.unbiased_exponent - self.fmt.mantissa_bits

        value = self.full_mantissa * self.sign_value
        value = torch.ldexp(value.to(self.fmt.value_dtype), scale)

        value = torch.where(self.is_inf, self.sign_value.to(self.fmt.value_dtype) * float("inf"), value)
        value = torch.where(self.is_nan, float("nan"), value)

        return value

    # --- Raw field extraction ---

    @property
    def sign_field(self) -> Tensor:
        """Extract the sign bit field (0: positive, 1: negative)."""
        return self.payload.to(self.fmt.dtype) & self.fmt.sign_mask

    @property
    def exponent_field(self) -> Tensor:
        """Extract the raw exponent field (biased encoding)."""
        return self.payload.to(self.fmt.dtype) & self.fmt.exponent_mask

    @property
    def mantissa_field(self) -> Tensor:
        """Extract the raw mantissa field (without hidden bit)."""
        return self.payload.to(self.fmt.dtype) & self.fmt.mantissa_mask

    @property
    def unsigned_field(self) -> Tensor:
        """Extract the unsigned magnitude (exponent + mantissa)."""
        return self.payload.to(self.fmt.dtype) & self.fmt.unsigned_mask

    # --- Semantic views ---

    @property
    def sign_bool(self) -> Tensor:
        """Compute the sign."""
        return self.sign_field != 0

    @property
    def sign_value(self) -> Tensor:
        """Compute the sign value (-1/+1)."""
        return self.sign_bool.to(self.fmt.dtype) * (-2) + 1

    @property
    def adjusted_exponent(self) -> Tensor:
        """Biased exponent value, treating subnormals as exponent 1."""
        exponent = self.exponent_field >> self.fmt.mantissa_bits
        # subnormal: if raw exponent is 0, effective biased exponent is 1
        return torch.where(exponent == 0, 1, exponent)

    @property
    def unbiased_exponent(self) -> Tensor:
        """Compute the actual (unbiased) exponent value."""
        return self.adjusted_exponent - self.fmt.exponent_bias

    @property
    def full_mantissa(self) -> Tensor:
        """Unsigned mantissa with implicit leading bit."""
        # Normalized: 1.mantissa, Subnormal: 0.mantissa
        is_normalized = self.exponent_field != 0
        hidden_bit = is_normalized.to(self.fmt.dtype) << self.fmt.mantissa_bits
        return hidden_bit | self.mantissa_field

    # --- Bit-pattern transforms ---

    @property
    def truncated(self) -> Tensor:
        """Bit pattern with inf/nan clamped to the largest finite value."""
        return torch.where(self.is_special, self.sign_field | self.fmt.max_unsigned, self.payload.to(self.fmt.dtype))

    @property
    def decremented(self) -> Tensor:
        """Bit pattern decremented by one ULP toward zero."""
        can_decrement = self.unsigned_field != 0
        return self.payload.to(self.fmt.dtype) - can_decrement

    @property
    def incremented(self) -> Tensor:
        """Bit pattern incremented by one ULP away from zero."""
        can_increment = self.unsigned_field <= self.fmt.max_unsigned
        return self.payload.to(self.fmt.dtype) + can_increment

    # --- Special-value predicates ---

    @property
    def is_finite(self) -> Tensor:
        """Check if values are finite (not NaN or Inf)."""
        return self.unsigned_field <= self.fmt.max_unsigned

    @property
    def is_special(self) -> Tensor:
        """Check if values are special (NaN or Inf)."""
        return self.unsigned_field > self.fmt.max_unsigned

    @property
    def is_inf(self) -> Tensor:
        """Check if values are infinite."""
        return self.unsigned_field == self.fmt.inf_unsigned

    @property
    def is_nan(self) -> Tensor:
        """Check if values are NaN."""
        return self.unsigned_field > self.fmt.inf_unsigned


# --- Internal implementation ---


def parts_to_float_binary(
    sign_bool: Tensor,
    exponent: Tensor,
    mantissa: Tensor,
    fmt: FloatFormat,
    *,
    rounding: RoundingMode,
    inf_mask: Tensor | None,
    nan_mask: Tensor | None,
) -> Tensor:
    """Construct floating-point binary bit patterns from decomposed parts.

    The represented value is (-1)^sign * 2^exponent * mantissa.

    Args:
        sign_bool: Boolean tensor, False=positive, True=negative.
        exponent: Integer tensor, unbiased exponent.
        mantissa: Integer tensor, unsigned mantissa (integer form).
        fmt: Target floating-point format.
        rounding: Rounding mode for mantissa truncation.
        inf_mask: Inf mask.
        nan_mask: NaN mask.

    Returns:
        Binary payload tensor in `fmt` integer layout.

    """

    def _find_msb_position(x: Tensor, width: int) -> Tensor:
        """Return the most-significant-bit index (0-based)."""
        pos = torch.zeros_like(x)
        remaining = x
        for shift in [width >> i for i in range(1, width.bit_length())]:
            high = remaining >> shift
            has_high = high != 0
            pos = pos + has_high.to(x.dtype) * shift
            remaining = torch.where(has_high, high, remaining)
        return pos

    if (
        sign_bool.shape != mantissa.shape
        or exponent.shape != mantissa.shape
        or (inf_mask is not None and inf_mask.shape != mantissa.shape)
        or (nan_mask is not None and nan_mask.shape != mantissa.shape)
    ):
        raise ValueError(
            f"Input tensors must have the same shape. Got sign_bool: {sign_bool.shape}, exponent: {exponent.shape}, mantissa: {mantissa.shape}"
            f", inf_mask: {inf_mask.shape if inf_mask is not None else None}, nan_mask: {nan_mask.shape if nan_mask is not None else None}"
        )

    # --- 1. Validate inputs and normalize dtypes ---

    bit_width = mantissa.element_size() * 8

    to_zero = mantissa == 0

    msb_position = _find_msb_position(mantissa, bit_width)

    biased_exp = exponent.to(fmt.dtype) + msb_position + fmt.exponent_bias

    # --- 2. Align mantissa to target format window ---

    # normal case (biased_exp >= 1): as `01.xxxx * 2^(E-bias)`
    # subnormal case (biased_exp < 1): as `00.xxxx * 2^(1-bias)`

    rshift = msb_position - fmt.mantissa_bits + (1 - biased_exp).clamp_min(0)

    align_drop_shift = rshift.clamp(0, bit_width - 1)

    round_offset = round_offset_true_form(mantissa, sign_bool, align_drop_shift, rounding)

    mantissa = mantissa >> align_drop_shift
    mantissa = mantissa.to(fmt.dtype) << (-rshift).clamp_min(0)
    mantissa = torch.where(rshift >= bit_width, 0, mantissa)

    # --- 3. Assemble exponent and mantissa fields ---

    is_overflow = biased_exp > fmt.max_biased_exp

    exp_field = biased_exp.clamp(0, fmt.max_biased_exp)
    mantissa_field = mantissa & fmt.mantissa_mask

    unsigned_field = (exp_field << fmt.mantissa_bits) | mantissa_field

    # --- 4. Apply rounding carry to packed unsigned field ---

    # normal case w/o carry: `(E, [01].xxxx)` -> `(E, [01].xxxx)`, nothing to do.
    # normal case w/i carry: `(E, [01].1111)` -> `(E+1, [01].0000)`, carry propagated to exponent, nothing to do.
    # normal case overflow: `(110, [01].1111)` -> `(111, [01].0000)`, promoted to inf automatically, nothing to do.
    # subnormal case w/o carry: `(0, [00].xxxx)` -> `(0, [00].xxxx)`, nothing to do.
    # subnormal case w/i carry: `(0, [00].1111)` -> `(1, [01].xxxx)`, carry propagated to exponent, nothing to do.

    unsigned_field = unsigned_field + round_offset

    # --- 5. Override zero/inf/nan special cases ---

    # Zero
    unsigned_field = torch.where(to_zero, fmt.zero_unsigned, unsigned_field)

    # Inf
    if inf_mask is not None:
        is_overflow = is_overflow | inf_mask

    unsigned_field = torch.where(is_overflow, fmt.inf_unsigned, unsigned_field)

    # NaN
    if nan_mask is not None:
        unsigned_field = torch.where(nan_mask, fmt.nan_unsigned, unsigned_field)

    # full binary field
    binary_field = (sign_bool.to(fmt.dtype) << fmt.unsigned_bits) | unsigned_field

    return binary_field
