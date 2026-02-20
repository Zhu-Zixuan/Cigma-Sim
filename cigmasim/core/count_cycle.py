# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""PE cycle counting kernels."""

import torch
from torch import Tensor

from cigmasim.config import PEMappingOrder


def _apply_lane_sharing(
    count: Tensor,
    lane_sharing: tuple[tuple[int, ...], ...] | None,
) -> Tensor:
    """Apply lane-sharing constraints to lane cycle counts."""
    if not lane_sharing:
        return count

    for group in lane_sharing:
        if not group:
            return count

        lanes = list(group)
        size = len(lanes)

        # shape: [..., 1]
        total = count[..., lanes].sum(dim=-1, keepdim=True, dtype=torch.int32)
        avg = (total + size - 1) // size

        # shape: [LANE]
        mask = torch.zeros(count.size(-1), dtype=torch.bool, device=count.device)
        mask[lanes] = True

        # shape: [..., LANE]
        count = torch.where(mask, avg, count)

    return count


def count_cycles_bit_serial(
    data: Tensor,
    digit_per_data: int,
    digit_width: int,
    lane_count: int,
    window_size: int,
    mapping_order: PEMappingOrder,
    lane_sharing: tuple[tuple[int, ...], ...] | None,
) -> Tensor:
    """Count cycles for bit-serial PE execution."""
    # --- 1. Validate PE and window dimensions ---

    if lane_count <= 0:
        raise ValueError(f"lane_count ({lane_count}) <= 0")

    segment_len = data.size(-1)

    lane_count = min(segment_len, lane_count)

    # circuit constraint: full parallel utilization required (padding expected)
    if segment_len % lane_count != 0:
        raise ValueError(f"segment_len ({segment_len}) % lane_count ({lane_count}) != 0")

    data_per_lane = segment_len // lane_count

    digit_per_window = window_size if window_size > 0 else digit_per_data * segment_len

    # circuit constraint: windows are grouped to process data bits in parallel
    if digit_per_window < digit_per_data or digit_per_window % digit_per_data != 0:
        raise ValueError(
            f"digit_per_window ({digit_per_window}) < digit_per_data ({digit_per_data}) or digit_per_window ({digit_per_window}) % digit_per_data ({digit_per_data}) != 0"
        )

    data_per_window = min(data_per_lane, digit_per_window // digit_per_data)

    # circuit constraint: temporal tiling alignment
    if data_per_lane % data_per_window != 0:
        raise ValueError(f"data_per_lane ({data_per_lane}) % data_per_window ({data_per_window}) != 0")

    window_per_lane = data_per_lane // data_per_window

    # --- 2. Reshape input by mapping order ---

    # shape: [..., LEN]

    batch_dims = data.shape[:-1]

    match mapping_order:
        case PEMappingOrder.STRIP:
            # Strip mapping: interleaved values across windows
            # shape: [..., window_per_lane, data_per_window, lane_count]
            mapping_shape = (*batch_dims, window_per_lane, data_per_window, lane_count)
            window_dim = -2
            # make reducing dimensions adjacent for better performance
            # shape: [..., window_per_lane, lane_count, data_per_window]
            # data = data.transpose(-1, -2)
        case PEMappingOrder.BLOCK:
            # Block mapping: consecutive values go to same window
            # shape: [..., window_per_lane, lane_count, data_per_window]
            mapping_shape = (*batch_dims, window_per_lane, lane_count, data_per_window)
            window_dim = -1
        case _:
            raise ValueError(f"Unknown mapping order: {mapping_order}")

    # shape: [..., window_per_lane, data_per_window, lane_count]
    # shape: [..., window_per_lane, lane_count, data_per_window]
    data = data.reshape(mapping_shape)

    # --- 3. Count per-window lane work ---

    # shape: [..., window_per_lane, data_per_window, lane_count]
    # shape: [..., window_per_lane, lane_count, data_per_window]
    count = torch.zeros_like(data, dtype=torch.int32)

    for _ in range(digit_per_data):
        digit = data & 1
        count += digit.to(torch.int32)
        data = data >> digit_width

    # shape: [..., window_per_lane, lane_count]
    count = count.sum(dim=window_dim - 1, dtype=torch.int32)

    # --- 4. Apply lane sharing ---

    count = _apply_lane_sharing(count, lane_sharing)

    # --- 5. Reduce lanes (parallel) and windows (serial) ---

    # shape: [..., window_per_lane] -> [...]
    count = count.amax(dim=-1).sum(dim=-1, dtype=torch.int32)

    return count


def count_cycles_bit_parallel(
    data: Tensor,
    digit_per_data: int,
    digit_width: int,
    lane_count: int,
    window_size: int,
    mapping_order: PEMappingOrder,
    lane_sharing: tuple[tuple[int, ...], ...] | None,
) -> Tensor:
    """Count cycles for bit-parallel PE execution."""
    # --- 1. Validate PE and stream dimensions ---

    if lane_count <= 0:
        raise ValueError(f"lane_count ({lane_count}) <= 0")

    segment_len = data.size(-1)

    # circuit constraint: lanes are grouped to process data bits in parallel
    if lane_count < digit_per_data or lane_count % digit_per_data != 0:
        raise ValueError(
            f"lane_count ({lane_count}) < digit_per_data ({digit_per_data}) or lane_count ({lane_count}) % digit_per_data ({digit_per_data}) != 0"
        )

    parallel_streams = lane_count // digit_per_data

    # circuit constraint: full parallel utilization required (padding expected)
    if segment_len % parallel_streams != 0:
        raise ValueError(f"segment_len ({segment_len}) % parallel_streams ({parallel_streams}) != 0")

    depth_per_stream = segment_len // parallel_streams

    data_per_window = min(depth_per_stream, window_size) if window_size > 0 else depth_per_stream

    # circuit constraint: temporal tiling alignment
    if depth_per_stream % data_per_window != 0:
        raise ValueError(f"depth_per_stream ({depth_per_stream}) % digit_per_window ({data_per_window}) != 0")

    window_per_lane = depth_per_stream // data_per_window

    # --- 2. Reshape input by mapping order ---

    # shape: [..., LEN]

    batch_dims = data.shape[:-1]

    match mapping_order:
        case PEMappingOrder.STRIP:
            # Strip mapping: interleaved values across windows
            # shape: [..., window_per_lane, data_per_window, parallel_streams]
            mapping_shape = (*batch_dims, window_per_lane, data_per_window, parallel_streams)
            window_dim = -2
        case PEMappingOrder.BLOCK:
            # Block mapping: consecutive values go to same window
            # shape: [..., window_per_lane, parallel_streams, data_per_window]
            mapping_shape = (*batch_dims, window_per_lane, parallel_streams, data_per_window)
            window_dim = -1
        case _:
            raise ValueError(f"Unknown mapping order: {mapping_order}")

    # shape: [..., window_per_lane, data_per_window, parallel_streams]
    # shape: [..., window_per_lane, parallel_streams, data_per_window]
    data = data.reshape(mapping_shape)

    # --- 3. Count bit-lane workload in each window ---

    # shape: [digit_per_data]
    positions = torch.arange(0, digit_per_data * digit_width, digit_width, dtype=data.dtype, device=data.device)

    # shape: [..., window_per_lane, data_per_window, parallel_streams, digit_per_data]
    # shape: [..., window_per_lane, parallel_streams, data_per_window, digit_per_data]
    digits = (data.unsqueeze(-1) >> positions.unsqueeze(0)) & 1

    # shape: [..., window_per_lane, parallel_streams, digit_per_data] -> [..., window_per_lane, lane_count]
    count = digits.sum(dim=window_dim - 1, dtype=torch.int32).flatten(start_dim=-2)

    # --- 4. Apply lane sharing ---

    count = _apply_lane_sharing(count, lane_sharing)

    # --- 5. Reduce lanes (parallel) and windows (serial) ---

    # shape: [..., window_per_lane] -> [...]
    count = count.amax(dim=-1).sum(dim=-1, dtype=torch.int32)

    return count
