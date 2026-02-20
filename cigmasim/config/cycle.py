# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""PE-level cycle counting configuration."""

from enum import StrEnum, auto
from typing import NamedTuple

from cigmasim.type import Encoding


class PEProcessingStrategy(StrEnum):
    """Processing strategies for PE-level bit processing.

    Attributes:
        DATA_LEVEL: No bit-level decomposition; data-level processing only.
        W_BIT_SERIAL: Weight bits are processed serially, one digit at a time.
        W_BIT_PARALLEL: Weight bits are processed in parallel across lanes.
        BOTH_BIT_SERIAL: Both activation and weight bits are processed serially.

    """

    DATA_LEVEL = auto()
    W_BIT_SERIAL = auto()
    W_BIT_PARALLEL = auto()
    BOTH_BIT_SERIAL = auto()


class PEMappingOrder(StrEnum):
    """Data-to-lane mapping orders within a PE.

    Attributes:
        BLOCK: Fill lanes first with contiguous data (block mapping).
        STRIP: Fill windows first with interleaved data (strip mapping).

    """

    BLOCK = auto()
    STRIP = auto()


class CycleCountingConfig(NamedTuple):
    """Configuration for PE-level cycle counting.

    Attributes:
        a_encode: Bit encoding for activations.
        w_encode: Bit encoding for weights.
        enable_value_skip: Whether to skip zero-valued activations.
        method: PE processing strategy (bit-serial, bit-parallel, etc.).
        mapping_order: Data-to-lane mapping order.
        lane_count: Number of parallel lanes in the PE.
        window_size: Number of digit positions processed per window.
            Non-positive means unbounded (best-case).
        lane_sharing: Optional lane sharing groups for load balancing.
            Each inner tuple lists lane indices that share work.

    """

    a_encode: Encoding
    w_encode: Encoding

    enable_value_skip: bool

    method: PEProcessingStrategy
    mapping_order: PEMappingOrder

    lane_count: int
    window_size: int

    lane_sharing: tuple[tuple[int, ...], ...] | None = None
