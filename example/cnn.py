# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Shared runner for CNN profiling experiments."""

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from tqdm import tqdm

from .common import load_dataset, make_log_name, parse_args, redirect_stderr_to_file, setup_experiment_logger
from .presets import QUANT_ARCH_NAMES, get_quant_cycle_fn, get_quant_value_fn
from .profiler import ChunkConfig, Profiler


def run_cnn_experiment(
    model_name: str,
    load_model: Callable[[torch.device], tuple[nn.Module, Any]],
    *,
    chunk_configs: dict[str, ChunkConfig] | None = None,
) -> None:
    """Run one CNN profiling experiment from CLI arguments."""
    # --- parse args ---

    args = parse_args(f"{model_name} profiling", QUANT_ARCH_NAMES)

    num_threads: int | None = args.num_threads
    device = torch.device(args.device)
    sim_cycle: bool = args.cycle
    sim_value: bool = args.value
    compile_only: bool = args.compile_only
    arch: str = args.arch
    batch_size: int = args.batch_size
    data: str = args.data
    split: str = args.split
    log_level: Literal["DEBUG", "INFO"] = args.log_level

    # --- initial setup ---

    if num_threads is not None:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)

    sims: list[str] = []
    if sim_cycle:
        sims.append("cycle")
    if sim_value:
        sims.append("value")
    if not sims:
        sims.append("none")

    log_name = make_log_name(model_name, sims, arch)

    log_file = Path(f"log/{log_name}.log")
    stderr_file = Path(f"compile.log/{log_name}.log")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)

    setup_experiment_logger(log_file, log_level)

    logger = logging.getLogger(model_name)

    # --- report setup ---

    logger.info("=" * 80)
    logger.info("Experiment : %s", model_name)
    logger.info("Arch       : %s", arch)
    logger.info("Simulation : %s", ", ".join(sims))
    logger.info("Device     : %s", device)
    if num_threads is not None:
        logger.info("Threads    : %d", num_threads)
    logger.info("Batch size : %d", batch_size)
    if chunk_configs:
        logger.info("Chunk cfgs : %d", len(chunk_configs))
    logger.info("=" * 80)

    # --- load model and dataset ---

    model, transform = load_model(device)
    logger.info("Model loaded: %s", model_name)

    dataset = load_dataset(data, split)
    logger.info("Dataset loaded: %s[%s] (%d samples)", data, split, len(dataset))

    # --- setup profiler ---

    profiler = Profiler(
        arch=arch,
        cycle_fn=get_quant_cycle_fn(arch) if sim_cycle else None,
        value_fn=get_quant_value_fn(arch) if sim_value else None,
        chunk_configs=chunk_configs,
    )
    profiler.register(model)
    logger.info("Profiler registered (%d hooks)", len(profiler._hooks))

    # --- run inference ---

    logger.info("Starting inference (%d batches) ...", (len(dataset) + batch_size - 1) // batch_size)
    with torch.no_grad(), redirect_stderr_to_file(str(stderr_file)):
        for i in tqdm(
            range(0, len(dataset) if not compile_only else batch_size, batch_size),
            desc=model_name,
            file=sys.stdout,
        ):
            batch = dataset[i : i + batch_size]
            images = [img.convert("RGB") for img in batch["image"]]
            inputs = torch.stack([transform(img) for img in images]).to(device)
            model(inputs)

    # --- report result ---

    if sim_cycle:
        logger.info("\n%s", profiler.speedup_summary())
    if sim_value:
        logger.info("\n%s", profiler.accuracy_summary())

    # --- clean up ---

    profiler.remove_hooks()
