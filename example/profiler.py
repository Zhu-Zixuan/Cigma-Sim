# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Hook-based GEMM profiler for cycle counting and accuracy evaluation."""

import logging
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from .presets import CycleFn, ValueFn

logger = logging.getLogger("Profiler")


class ChunkConfig(NamedTuple):
    """Chunking strategy for one GEMM layer.

    Each field pair controls chunking along one matrix dimension.
    Set `*_chunk_dim` to `None` to skip chunking for that matrix.

    Attributes:
        a_chunk_dim: Dimension to split `a` along, or `None`.
        a_chunk_size: Max elements per `a` chunk along `a_chunk_dim`.
        wt_chunk_dim: Dimension to split `wt` along, or `None`.
        wt_chunk_size: Max elements per `wt` chunk along `wt_chunk_dim`.

    """

    a_chunk_dim: int | None = None
    a_chunk_size: int = 0

    wt_chunk_dim: int | None = None
    wt_chunk_size: int = 0


@dataclass
class LayerResult:
    """Accumulated profiling statistics for one layer."""

    name: str
    type: str

    mac_op: int = 0
    cycle: int = 0

    mismatch: int = 0
    total_are: float = 0
    result_count: int = 0


class Profiler:
    """Hook-based profiler for PyTorch models."""

    def __init__(
        self,
        arch: str,
        cycle_fn: CycleFn | None = None,
        value_fn: ValueFn | None = None,
        chunk_configs: dict[str, ChunkConfig] | None = None,
    ) -> None:
        """Initialize profiler.

        Args:
            arch: Architecture name for display and report.
            cycle_fn: Cycle simulation function `(a, wt) -> Tensor`.
            value_fn: Value simulation function `(a, wt) -> Tensor`.
            chunk_configs: Optional per-layer chunk settings.

        """
        self.arch = arch
        self.cycle_fn = cycle_fn
        self.value_fn = value_fn
        self.chunk_configs: dict[str, ChunkConfig] = chunk_configs or {}
        self.results: OrderedDict[str, LayerResult] = OrderedDict()
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def register(self, model: nn.Module) -> None:
        """Register forward hooks on target layers."""
        self.remove_hooks()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

        logger.debug("Registered %d hooks", len(self._hooks))

    def _make_hook(self, name: str) -> Callable:
        def hook(
            module: nn.Module,
            inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> None:
            x = inputs[0].detach()
            if isinstance(module, nn.Linear):
                a, wt = extract_gemm_linear(module, x)
            elif isinstance(module, nn.Conv2d):
                a, wt = extract_gemm_conv2d(module, x)
            else:
                return
            self._on_gemm(name, type(module).__name__, a, wt)

        return hook

    def _on_gemm(
        self,
        name: str,
        layer_type: str,
        a: Tensor,
        wt: Tensor,
    ) -> None:
        """Profile one GEMM call and update layer statistics."""
        # --- update MACs on original shapes ---
        macs = compute_macs(a, wt)
        if name not in self.results:
            self.results[name] = LayerResult(name, layer_type)
        self.results[name].mac_op += macs

        # --- profile chunks ---
        is_chunked = self._find_chunk_config(name) is not None
        for a_chunk, wt_chunk in self._iter_chunks(name, a, wt):
            self._log_tensor_shape(name, layer_type, a_chunk, wt_chunk)
            self._profile_cycle(name, layer_type, a_chunk, wt_chunk)
            self._profile_value(name, layer_type, a_chunk, wt_chunk)
        if is_chunked and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self) -> None:
        """Clear all accumulated per-layer statistics."""
        self.results.clear()

    def speedup_summary(self) -> str:
        """Build cycle summary table."""
        lines: list[str] = []

        header = f"{'Layer':<50} | {'Type':<15} | {'MACs':>20} | {'Cycles':>20} | {'Speedup':>10}"
        lines.append(header)
        lines.append("-" * len(header))

        total_macs = 0
        total_cycles = 0

        for r in self.results.values():
            total_macs += r.mac_op
            total_cycles += r.cycle
            speedup = r.mac_op / r.cycle if r.cycle > 0 else float("inf")
            line = f"{r.name:<50} | {r.type:<15} | {r.mac_op:>20,} | {r.cycle:>20,} | {speedup:>10.4f}"
            lines.append(line)

        lines.append("=" * len(header))
        total_speedup = total_macs / total_cycles if total_cycles > 0 else float("inf")
        total_line = (
            f"{self.arch + ' TOTAL':<50} | {'':<15} | {total_macs:>20,} | {total_cycles:>20,} | {total_speedup:>10.4f}"
        )
        lines.append(total_line)

        return "\n".join(lines)

    def accuracy_summary(self) -> str:
        """Build value-accuracy summary table."""
        lines: list[str] = []

        header = f"{'Layer':<50} | {'Type':<15} | {'Results':>20} | {'Mismatches':>20} | {'Mean ARE':>10}"
        lines.append(header)
        lines.append("-" * len(header))

        total_mismatch: int = 0
        total_are: float = 0.0
        total_count: int = 0

        for r in self.results.values():
            total_count += r.result_count
            total_mismatch += r.mismatch
            total_are += r.total_are
            mean_are = r.total_are / max(r.mismatch, 1)
            line = f"{r.name:<50} | {r.type:<15} | {r.result_count:>20,} | {r.mismatch:>20,} | {mean_are:>10.4e}"
            lines.append(line)

        lines.append("=" * len(header))
        mean_are = total_are / max(total_mismatch, 1)
        total_line = (
            f"{self.arch + ' TOTAL':<50} | {'':<15} | {total_count:>20,} | {total_mismatch:>20,} | {mean_are:>10.4e}"
        )
        lines.append(total_line)

        return "\n".join(lines)

    # --- chunk helpers ---

    def _find_chunk_config(self, name: str) -> ChunkConfig | None:
        """Resolve chunk config by exact-name or glob-pattern match."""
        cfg = self.chunk_configs.get(name)
        if cfg is not None:
            return cfg
        for pattern, cfg in self.chunk_configs.items():
            if fnmatch(name, pattern):
                return cfg
        return None

    @staticmethod
    def _iter_dim(
        tensor: Tensor,
        chunk_dim: int | None,
        chunk_size: int,
    ) -> Iterator[Tensor]:
        """Iterate tensor slices along one dimension."""
        if chunk_dim is None or chunk_size <= 0:
            yield tensor
            return
        dim = chunk_dim % tensor.ndim
        size = tensor.shape[dim]
        for start in range(0, size, chunk_size):
            yield tensor.narrow(dim, start, min(chunk_size, size - start))

    def _iter_chunks(
        self,
        name: str,
        a: Tensor,
        wt: Tensor,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield `(a_chunk, wt_chunk)` pairs for one layer call."""
        cfg = self._find_chunk_config(name)
        if cfg is None:
            yield a, wt
            return

        logger.debug(
            "[chunk] %s | a: dim=%s chunk=%d | wt: dim=%s chunk=%d",
            name,
            cfg.a_chunk_dim,
            cfg.a_chunk_size,
            cfg.wt_chunk_dim,
            cfg.wt_chunk_size,
        )

        for a_c in self._iter_dim(a, cfg.a_chunk_dim, cfg.a_chunk_size):
            for wt_c in self._iter_dim(wt, cfg.wt_chunk_dim, cfg.wt_chunk_size):
                yield a_c, wt_c

    # --- core profiling ---

    def _log_tensor_shape(
        self,
        name: str,
        layer_type: str,
        a: Tensor,
        wt: Tensor,
    ) -> None:
        """Log one GEMM chunk shape for debugging."""
        logger.debug(
            "[shape] %s (%s) | a: %s (%s) | wt: %s (%s) | broadcast: %s",
            name,
            layer_type,
            list(a.shape),
            a.dtype,
            list(wt.shape),
            wt.dtype,
            wt.shape[-2] * a.shape[-2] * a.shape[-1],
        )

    def _profile_cycle(
        self,
        name: str,
        layer_type: str,
        a: Tensor,
        wt: Tensor,
    ) -> None:
        """Run cycle simulation for one GEMM chunk."""
        if self.cycle_fn is None:
            return

        c = self.cycle_fn(a, wt)
        cycles = int(c.sum().item())

        if name not in self.results:
            self.results[name] = LayerResult(name, layer_type)

        self.results[name].cycle += cycles

    def _profile_value(
        self,
        name: str,
        layer_type: str,
        a: Tensor,
        wt: Tensor,
    ) -> None:
        """Run value simulation and accumulate accuracy metrics."""
        if self.value_fn is None:
            return

        golden = a @ wt.transpose(-1, -2)

        simulated = self.value_fn(a, wt)

        delta = 0.0

        diff_abs = (golden - simulated).abs()
        are = diff_abs / golden.abs().clamp(min=1e-30)

        mismatch = int((diff_abs > delta).sum().item())
        total_are = float(are.sum().item())
        count = golden.numel()

        if name not in self.results:
            self.results[name] = LayerResult(name, layer_type)

        r = self.results[name]
        r.mismatch += mismatch
        r.total_are += total_are
        r.result_count += count


# --- GEMM extraction ---


def extract_gemm_linear(module: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor]:
    """Convert `nn.Linear` input/weight to batched GEMM operands."""
    b = x.shape[0]
    k = x.shape[-1]
    m = x[0].numel() // k

    a = x.reshape(b, m, k)
    wt = module.weight.detach().unsqueeze(0).expand(b, -1, -1)
    return a, wt


def extract_gemm_conv2d(module: nn.Conv2d, x: Tensor) -> tuple[Tensor, Tensor]:
    """Convert `nn.Conv2d` input/weight to batched GEMM operands."""
    b = x.shape[0]
    g = module.groups
    c_out = module.out_channels
    c_in_per_g = module.in_channels // g
    kh, kw = module.kernel_size
    k = c_in_per_g * kh * kw
    n = c_out // g

    padding = module.padding
    if not isinstance(padding, tuple):
        msg = f"nn.functional.unfold requires tuple padding, got {padding!r}"
        raise TypeError(msg)

    # shape: [b, g*k, m]
    x_unfolded = nn.functional.unfold(
        x,
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=padding,
        stride=module.stride,
    )
    m = x_unfolded.shape[-1]

    # [b, g, k, m] -> permute -> [b*g, m, k]
    a = x_unfolded.reshape(b, g, k, m).permute(0, 1, 3, 2).reshape(b * g, m, k)

    # [c_out, c_in/g, kh, kw] -> [g, n, k] -> expand -> [b*g, n, k]
    wt = module.weight.detach().reshape(g, n, k)
    wt = wt.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * g, n, k)

    return a, wt


def compute_macs(a: Tensor, wt: Tensor) -> int:
    """Return total MAC count from GEMM operand shapes."""
    m, k = a.shape[-2], a.shape[-1]
    n = wt.shape[-2]
    batch = a[..., 0, 0].numel()
    return batch * m * n * k
