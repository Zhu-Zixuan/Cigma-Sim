# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Bit-level encoding schemes and signed-digit representations."""

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Self

from torch import Tensor

from .utils import logical_right_shift


class DigitType(StrEnum):
    """Digit sets for signed-digit encodings.

    Attributes:
        TERNARY: Digits from {-1, 0, +1}, 1 bit per position.
        QUINARY: Digits from {-2, -1, 0, +1, +2}, 2 bits per position.

    """

    TERNARY = auto()
    QUINARY = auto()

    @property
    def digit_width(self) -> int:
        """Bits required per digit position."""
        _digit_width: dict[DigitType, int] = {
            DigitType.TERNARY: 1,
            DigitType.QUINARY: 2,
        }
        return _digit_width[self]


class Encoding(StrEnum):
    """Encoding schemes for signed integers.

    Attributes:
        TWOS_COMPLEMENT: Standard two's complement representation.
        SIGN_MAGNITUDE: Sign bit plus unsigned magnitude.
        BOOTH_RADIX2: Radix-2 Booth recoding (ternary digits: -1, 0, +1).
        BOOTH_RADIX4: Radix-4 Booth recoding (quinary digits: -2, -1, 0, +1, +2).
        CSD_CANONICAL: Canonical Signed Digit (non-adjacent form).
        CSD_COMPACT: Compact CSD using radix-4 quinary digits.

    """

    TWOS_COMPLEMENT = auto()
    SIGN_MAGNITUDE = auto()
    BOOTH_RADIX2 = auto()
    BOOTH_RADIX4 = auto()
    CSD_CANONICAL = auto()
    CSD_COMPACT = auto()

    @property
    def digit_type(self) -> DigitType:
        """Digit type for this encoding."""
        _digit_type: dict[Encoding, DigitType] = {
            Encoding.TWOS_COMPLEMENT: DigitType.TERNARY,
            Encoding.SIGN_MAGNITUDE: DigitType.TERNARY,
            Encoding.BOOTH_RADIX2: DigitType.TERNARY,
            Encoding.BOOTH_RADIX4: DigitType.QUINARY,
            Encoding.CSD_CANONICAL: DigitType.TERNARY,
            Encoding.CSD_COMPACT: DigitType.QUINARY,
        }
        return _digit_type[self]

    @property
    def digit_width(self) -> int:
        """Bits required per digit position."""
        return self.digit_type.digit_width

    def get_digits(self, bits: int) -> int:
        """Get number of digits to represent number of bits."""
        return (bits + self.digit_width - 1) // self.digit_width


@dataclass
class TernaryDigits:
    """Ternary signed-digit bit planes.

    Attributes:
        nonzero: Non-zero indicator bit plane.
        negative: Negative digit bit plane.
        positive: Positive digit bit plane.

    """

    nonzero: Tensor
    negative: Tensor
    positive: Tensor

    @property
    def bit_width(self) -> int:
        return self.nonzero.element_size() * 8

    # --- Value conversion ---

    @property
    def value(self) -> Tensor:
        """Return decoded signed integer values."""
        return self.positive - self.negative

    # --- Selection and predicates ---

    def select_masked(self, mask: Tensor) -> Self:
        """Select elements where `mask` is True."""
        return self.__class__(
            nonzero=self.nonzero.masked_select(mask),
            negative=self.negative.masked_select(mask),
            positive=self.positive.masked_select(mask),
        )

    @property
    def is_nonadjacent_form(self) -> Tensor:
        """Return whether digits satisfy non-adjacent form (NAF)."""
        this_digit_is_one = self.nonzero
        prev_digit_is_one = self.nonzero << 1
        adjacent = prev_digit_is_one & this_digit_is_one
        return adjacent == 0

    # --- Sign and bit transforms ---

    def apply_sign(self, sign_mask: Tensor) -> Self:
        """Flip signs at positions indicated by `sign_mask`."""
        swap = self.nonzero * sign_mask
        return self.__class__(
            nonzero=self.nonzero,
            negative=self.negative ^ swap,
            positive=self.positive ^ swap,
        )

    @property
    def negate(self) -> Self:
        """Return negated digits."""
        return self.__class__(
            nonzero=self.nonzero,
            negative=self.positive,
            positive=self.negative,
        )

    def mask_bits(self, mask: Tensor) -> Self:
        """Keep only bits indicated by `mask`."""
        return self.__class__(
            nonzero=self.nonzero & mask,
            negative=self.negative & mask,
            positive=self.positive & mask,
        )

    def shift_left(self, shifts: Tensor) -> Self:
        """Shift all bit planes left by `amount`."""
        return self.__class__(
            nonzero=self.nonzero << shifts,
            negative=self.negative << shifts,
            positive=self.positive << shifts,
        )

    def shift_right(self, shifts: Tensor) -> Self:
        """Shift all bit planes right by `amount`."""
        return self.__class__(
            nonzero=logical_right_shift(self.nonzero, shifts, self.bit_width),
            negative=logical_right_shift(self.negative, shifts, self.bit_width),
            positive=logical_right_shift(self.positive, shifts, self.bit_width),
        )


@dataclass
class QuinaryDigits:
    """Quinary signed-digit bit planes.

    Attributes:
        nonzero: Tensor where bit i is 1 if digit i is non-zero.
        negative: Tensor where bit i is 1 if digit i is negative.
        positive: Tensor where bit i is 1 if digit i is positive.
        doubled: Tensor where bit i is 1 if |digit i| = 2.

    """

    nonzero: Tensor
    negative: Tensor
    positive: Tensor
    doubled: Tensor

    @property
    def bit_width(self) -> int:
        return self.nonzero.element_size() * 8

    # --- Internal helpers ---

    def _to_ternary_mark(self, mark: Tensor) -> Tensor:
        """Expand quinary marks into ternary bit positions."""
        return ((self.doubled & mark) << 1) | (~self.doubled & mark)

    # --- Value conversion ---

    @property
    def value(self) -> Tensor:
        """Return decoded signed integer values."""
        return self._to_ternary_mark(self.positive) - self._to_ternary_mark(self.negative)

    @property
    def ternary(self) -> TernaryDigits:
        """Expand to ternary representation."""
        ternary_nonzero = self._to_ternary_mark(self.nonzero)
        ternary_negative = self._to_ternary_mark(self.negative)
        ternary_positive = self._to_ternary_mark(self.positive)
        return TernaryDigits(nonzero=ternary_nonzero, negative=ternary_negative, positive=ternary_positive)

    # --- Selection and predicates ---

    def select_masked(self, mask: Tensor) -> Self:
        """Select elements where `mask` is True."""
        return self.__class__(
            nonzero=self.nonzero.masked_select(mask),
            negative=self.negative.masked_select(mask),
            positive=self.positive.masked_select(mask),
            doubled=self.doubled.masked_select(mask),
        )

    @property
    def is_nonadjacent_form(self) -> Tensor:
        """Return whether digits satisfy quinary non-adjacent form."""
        this_digit_is_one = self.nonzero & ~self.doubled
        prev_digit_is_two = (self.nonzero & self.doubled) << 2
        adjacent = prev_digit_is_two & this_digit_is_one
        return adjacent == 0

    # --- Sign and bit transforms ---

    def apply_sign(self, sign_mask: Tensor) -> Self:
        """Flip signs at positions indicated by `sign_mask`."""
        swap = self.nonzero * sign_mask
        return self.__class__(
            nonzero=self.nonzero,
            negative=self.negative ^ swap,
            positive=self.positive ^ swap,
            doubled=self.doubled,
        )

    @property
    def negate(self) -> Self:
        """Return negated digits."""
        return self.__class__(
            nonzero=self.nonzero,
            negative=self.positive,
            positive=self.negative,
            doubled=self.doubled,
        )

    def mask_bits(self, mask: Tensor) -> Self:
        """Keep only bits indicated by `mask`."""
        return self.__class__(
            nonzero=self.nonzero & mask,
            negative=self.negative & mask,
            positive=self.positive & mask,
            doubled=self.doubled & mask,
        )

    def shift_left(self, shifts: Tensor) -> Self:
        """Shift all bit planes left by `amount`."""
        return self.__class__(
            nonzero=self.nonzero << shifts,
            negative=self.negative << shifts,
            positive=self.positive << shifts,
            doubled=self.doubled << shifts,
        )

    def shift_right(self, amount: Tensor) -> Self:
        """Shift all bit planes right by `amount`."""
        return self.__class__(
            nonzero=logical_right_shift(self.nonzero, amount, self.bit_width),
            negative=logical_right_shift(self.negative, amount, self.bit_width),
            positive=logical_right_shift(self.positive, amount, self.bit_width),
            doubled=logical_right_shift(self.doubled, amount, self.bit_width),
        )


def get_signed_digits(bits: Tensor, enc: Encoding) -> TernaryDigits | QuinaryDigits:
    """Convert two's complement integers to signed-digit bit planes.

    Args:
        bits: Integer tensor in two's complement.
        enc: Target encoding scheme.

    Returns:
        `TernaryDigits` for ternary encodings or `QuinaryDigits`
        for quinary encodings.

    Raises:
        ValueError: If `enc` is unsupported.

    """
    # --- 1. Resolve machine-word constants ---

    width = bits.element_size() * 8
    mask_0x55 = 0x5555555555555555 >> (64 - width)

    # --- 2. Encode by selected scheme ---

    match enc:
        case Encoding.TWOS_COMPLEMENT:
            sign_mask = 1 << (width - 1)
            nonzero = bits
            negative = bits & sign_mask
            positive = bits & (sign_mask - 1)
            return TernaryDigits(nonzero=nonzero, negative=negative, positive=positive)

        case Encoding.SIGN_MAGNITUDE:
            full_sign = bits >> (width - 1)
            value = bits + full_sign
            nonzero = value ^ full_sign
            negative = ~value & full_sign
            positive = value & ~full_sign
            return TernaryDigits(nonzero=nonzero, negative=negative, positive=positive)

        case Encoding.BOOTH_RADIX2:
            bits_lsh = bits << 1
            diff_lsh = bits ^ bits_lsh
            nonzero = diff_lsh
            negative = bits & ~bits_lsh
            positive = ~bits & bits_lsh
            return TernaryDigits(nonzero=nonzero, negative=negative, positive=positive)

        case Encoding.BOOTH_RADIX4:
            bits_lsh = bits << 1
            bits_rsh = bits >> 1
            diff_lsh = bits ^ bits_lsh
            diff_rsh = bits ^ bits_rsh
            nonzero = (diff_rsh | diff_lsh) & mask_0x55
            doubled = diff_rsh & ~diff_lsh & mask_0x55
            negative = ~(bits & bits_lsh) & bits_rsh & mask_0x55
            positive = (bits | bits_lsh) & ~bits_rsh & mask_0x55
            return QuinaryDigits(nonzero=nonzero, negative=negative, positive=positive, doubled=doubled)

        case Encoding.CSD_CANONICAL:
            bits_rsh = bits >> 1
            value = bits + bits_rsh
            nonzero = value ^ bits_rsh
            negative = ~value & bits_rsh
            positive = value & ~bits_rsh
            return TernaryDigits(nonzero=nonzero, negative=negative, positive=positive)

        case Encoding.CSD_COMPACT:
            bits_rsh = bits >> 1
            value = bits + bits_rsh
            csd_nonzero = value ^ bits_rsh
            csd_negative = ~value & bits_rsh
            csd_positive = value & ~bits_rsh
            nonzero = (csd_nonzero | (csd_nonzero >> 1)) & mask_0x55
            negative = (csd_negative | (csd_negative >> 1)) & mask_0x55
            positive = (csd_positive | (csd_positive >> 1)) & mask_0x55
            doubled = (csd_nonzero >> 1) & mask_0x55
            return QuinaryDigits(nonzero=nonzero, negative=negative, positive=positive, doubled=doubled)

        case _:
            raise ValueError(f"Unsupported encoding: {enc}")
