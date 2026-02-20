# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Shared helpers for experiment scripts."""

import argparse
import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from datasets import Dataset, load_from_disk

# --- project cache ---


WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
"""Project weights cache directory."""


# --- cli args ---


def parse_args(description: str, arch_names: list[str]) -> argparse.Namespace:
    """Parse shared CLI arguments for profiling scripts."""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--data", type=str, required=True, help="Dataset directory")
    parser.add_argument("--split", type=str, required=True, help="Dataset split")
    parser.add_argument("--arch", type=str, required=True, choices=arch_names, help="Architecture to simulate")
    parser.add_argument("--cycle", action="store_true", help="Run cycle simulation")
    parser.add_argument("--value", action="store_true", help="Run value simulation")
    parser.add_argument("--compile-only", action="store_true", help="Run compile only")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO"],
        help="Log level for file output (default: INFO)",
    )
    parser.add_argument("--num-threads", type=int, default=None)

    return parser.parse_args()


# --- dataset ---


def load_dataset(data_dir: str, split: str) -> Dataset:
    """Load a local HuggingFace dataset split."""
    path = Path(data_dir)
    if not path.is_dir():
        msg = f"Dataset directory not found: {data_dir}"
        raise FileNotFoundError(msg)
    full_ds = load_from_disk(str(path))
    ds = full_ds[split]
    return ds


# --- logging ---


def make_log_name(model_name: str, sims: list[str], arch: str) -> str:
    """Build experiment log filename stem."""
    return f"{model_name}_{'_'.join(sims)}_{arch}"


def setup_experiment_logger(
    log_file: Path,
    file_log_level: Literal["DEBUG", "INFO"] = "INFO",
) -> None:
    """Configure root logger with console and file handlers."""
    log_level = getattr(logging, file_log_level)

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # --- root ---

    root = logging.getLogger()
    root.handlers.clear()
    root_level = min(logging.INFO, log_level) if log_file else logging.INFO
    root.setLevel(root_level)

    # --- console ---

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # --- file ---

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, "w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


@contextmanager
def redirect_stderr_to_file(log_file: str) -> Generator[None, None, None]:
    """Temporarily redirect stderr to a file."""
    log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    sys_stderr_fd = os.dup(sys.stderr.fileno())
    try:
        sys.stderr.flush()
        os.dup2(log_fd, sys.stderr.fileno())
        yield
    finally:
        sys.stderr.flush()
        os.dup2(sys_stderr_fd, sys.stderr.fileno())
        os.close(sys_stderr_fd)
        os.close(log_fd)
