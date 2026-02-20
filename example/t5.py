# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""T5-small command-line entry for simulation profiling.

Example:
    python -m tc2025.t5 --data datasets/wmt14/de-en --split test --device cuda --cycle --batch-size 1 --arch cigma
    python -m tc2025.t5 --data datasets/wmt14/de-en --split test --device cuda --value --batch-size 1 --arch bitlet

"""

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase, T5ForConditionalGeneration, T5Tokenizer

from .common import WEIGHTS_DIR
from .profiler import ChunkConfig
from .transformer import run_transformer_experiment

_MODEL_NAME = "google-t5/t5-small"
_CHUNK_CONFIGS: dict[str, ChunkConfig] = {
    "lm_head": ChunkConfig(wt_chunk_dim=-2, wt_chunk_size=4096),
}

_TASK_PREFIX = "translate English to Germany: "
_MAX_SOURCE_LENGTH = 512
_MAX_TARGET_LENGTH = 128


def _load_model(device: torch.device) -> tuple[nn.Module, PreTrainedTokenizerBase]:
    """Load pretrained T5 model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained(_MODEL_NAME, cache_dir=str(WEIGHTS_DIR))
    tokenizer = T5Tokenizer.from_pretrained(_MODEL_NAME, cache_dir=str(WEIGHTS_DIR))
    model = model.to(device).eval()  # type: ignore[arg-type]
    return model, tokenizer


def _prepare_inputs(
    batch: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> dict[str, Tensor]:
    """Tokenize a WMT14 batch and return model kwargs."""
    translations = batch["translation"]
    input_sequences = [e["en"] for e in translations]

    encoding = tokenizer(
        [_TASK_PREFIX + seq for seq in input_sequences],
        padding="longest",
        max_length=_MAX_SOURCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    target_encoding = tokenizer(
        [e["de"] for e in translations],
        padding="longest",
        max_length=_MAX_TARGET_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": encoding.input_ids.to(device),
        "attention_mask": encoding.attention_mask.to(device),
        "labels": labels.to(device),
    }


def main() -> None:
    """Run T5 profiling experiment."""
    run_transformer_experiment(_MODEL_NAME, _load_model, _prepare_inputs, chunk_configs=_CHUNK_CONFIGS)


if __name__ == "__main__":
    main()
