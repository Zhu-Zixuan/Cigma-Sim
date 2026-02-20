# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Quantized-integer format and bit-pattern wrapper definition."""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from .dtype import get_compute_integer_dtype
from .rounding import RoundingMode, round_offset_complement
from .utils import build_segment


@dataclass(frozen=True)
class QuantFormat:
    """Quantized-integer format definition.

    Attributes:
        bits: Number of bits of the payload.
        frac: Number of bits of the fraction part.

    """

    bits: int
    frac: int

    @property
    def max_value(self) -> int:
        """Get maximum value."""
        return (1 << (self.bits - 1)) - 1

    @property
    def min_value(self) -> int:
        """Get minimum value."""
        return -(1 << (self.bits - 1))

    @property
    def frac_scale(self) -> int:
        """Get fraction scale."""
        return 1 << self.frac

    # --- Dtype helpers ---

    @property
    def dtype(self) -> torch.dtype:
        """Get compute dtype for this format."""
        return get_compute_integer_dtype(self.bits)


@dataclass
class QuantDataWrap:
    """Wrapper for quantized fixed-point values.

    Attributes:
        payload: Integer tensor holding quantized values.
        fmt: Quantized-integer format specification.

    """

    payload: Tensor
    fmt: QuantFormat

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
        """Forward to `self.payload` and wrap the result."""
        payload = self.payload.unsqueeze(dim)
        return self.__class__(payload, self.fmt)

    def broadcast_to(self, shape: torch.Size) -> Self:
        """Forward to `self.payload` and wrap the result."""
        payload = self.payload.broadcast_to(shape)
        return self.__class__(payload, self.fmt)

    def select(self, dim: int, index: int) -> Self:
        """Forward to `self.payload` and wrap the result."""
        payload = self.payload.select(dim, index)
        return self.__class__(payload, self.fmt)

    def clone(self) -> Self:
        """Forward to `self.payload` and wrap the result."""
        payload = self.payload.clone()
        return self.__class__(payload, self.fmt)

    def split(self, split_size: int, dim: int) -> tuple[Self, ...]:
        """Forward to `self.payload` and wrap the result."""
        chunks = self.payload.split(split_size, dim)
        return tuple(self.__class__(chunk, self.fmt) for chunk in chunks)

    # --- Build segment ---

    def build_segment(self, segment_len: int) -> Self:
        """Split the last dimension into fixed-size segments."""
        payload = build_segment(self.payload, segment_len)
        return self.__class__(payload, self.fmt)

    # --- Format conversion ---

    @classmethod
    def from_value(cls, value: Tensor, fmt: QuantFormat) -> Self:
        """Construct quantized-integer binary bit pattern from value.

        Args:
            value: Source value tensor.
            fmt: Target quantized format.

        Returns:
            Wrapped quantized payload.

        """
        value = value * fmt.frac_scale
        value = value.clamp(fmt.min_value, fmt.max_value).to(fmt.dtype)
        return cls(value, fmt)

    def to_value(self) -> Tensor:
        """Decode quantized-integer values from bit patterns."""
        value = self.payload.to(torch.float32) / self.fmt.frac_scale
        return value


def clog2(x: int) -> int:
    """Compute ceil(log2(x))."""
    return (x - 1).bit_length()


def keep_fraction(
    data: Tensor,
    fmt: QuantFormat,
    new_frac: int,
    *,
    mode: RoundingMode = RoundingMode.FULL_DOWN,
) -> tuple[Tensor, QuantFormat]:
    """Adjust fractional precision to `frac` with optional extension."""
    if new_frac < fmt.frac:
        drop_width = fmt.frac - new_frac

        if drop_width >= fmt.bits:
            raise ValueError(
                f"drop_width {drop_width} >= fmt.bits {fmt.bits}, fmt.frac: {fmt.frac}, new_frac: {new_frac}"
            )

        new_fmt = QuantFormat(fmt.bits - drop_width, new_frac)
        new_data = (data >> drop_width) + round_offset_complement(data, drop_width, mode).to(data.dtype)
        new_data = new_data.to(new_fmt.dtype)

        return new_data, new_fmt

    if new_frac > fmt.frac:
        add_width = new_frac - fmt.frac

        new_fmt = QuantFormat(fmt.bits + add_width, new_frac)
        new_data = data.to(new_fmt.dtype) << add_width

        return new_data, new_fmt

    return data, fmt


def keep_width(
    data: Tensor,
    fmt: QuantFormat,
    width: int,
) -> tuple[Tensor, QuantFormat]:
    """Adjust signed width to `width` with optional extension."""
    if 0 < width < fmt.bits:
        new_fmt = QuantFormat(width, fmt.frac)

        shift = torch.iinfo(new_fmt.dtype).bits - width
        new_data = (data.to(new_fmt.dtype) << shift) >> shift

        return new_data, new_fmt

    if width > fmt.bits:
        new_fmt = QuantFormat(width, fmt.frac)
        new_data = data.to(new_fmt.dtype)

        return new_data, new_fmt

    return data, fmt


def check_overflow(
    data: Tensor,
    width: int,
) -> Tensor:
    """Check whether values overflow a signed `width`-bit range."""
    max_value = (1 << (width - 1)) - 1
    min_value = -(1 << (width - 1))
    return (data > max_value) | (data < min_value)
