# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""ResNet-152 command-line entry for simulation profiling.

Example:
    python -m tc2025.resnet152 --data datasets/Flickr8k --split test --device cuda --cycle --arch cigma
    python -m tc2025.resnet152 --data datasets/Flickr8k --split test --device cuda --cycle --arch pragmatic

"""

import torch
import torchvision.models as models
from torch import nn

from .cnn import run_cnn_experiment
from .common import WEIGHTS_DIR

_MODEL_NAME = "torchvision.models/resnet152"


def _load_model(device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Load pretrained ResNet-152 and its image transforms."""
    torch.hub.set_dir(str(WEIGHTS_DIR))
    weights = models.ResNet152_Weights.IMAGENET1K_V1
    model = models.resnet152(weights=weights).to(device).eval()
    transform = weights.transforms()
    return model, transform


def main() -> None:
    """Run ResNet-152 profiling experiment."""
    run_cnn_experiment(_MODEL_NAME, _load_model)


if __name__ == "__main__":
    main()
