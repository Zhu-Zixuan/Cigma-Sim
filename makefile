# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

# Cigma-Sim Makefile


PYTHON ?= python


PROJ_ROOT := $(shell pwd)
PROJ_NAME := cigma-sim


.DEFAULT_GOAL := help


include scripts/container.mk
include scripts/dev.mk


# --- Help Info ---

.PHONY: help
help: ## Show available targets and their help messages
	@echo "Usage: make <target>"
	@awk 'BEGIN {FS = ": .*## "; i = 0; max_len = 0; file = ""} \
		/^[a-zA-Z0-9_-]+:.*?## / { \
			targets[i] = $$1; help_msgs[i] = $$2; files[i] = FILENAME; len = length($$1); \
			if (len > max_len) max_len = len; \
			i++; \
		} END { \
			for (j = 0; j < i; j++) { \
				if (files[j] != file) { \
					file = files[j]; \
					# printf "\n\033[33m%s:\033[0m\n", file; \
					printf "\n"; \
				} \
				printf "  \033[36m%-" max_len "s\033[0m %s\n", targets[j], help_msgs[j]; \
			} \
		}' $(MAKEFILE_LIST)


# --- Cleaning ---

.PHONY: clean
clean: ## Clean up cache and temporary files
	rm -rf .*_cache *.log site .coverage htmlcov log
	rm -rf ~/.triton/cache /tmp/torchinductor_*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +


# --- Demo ---

TORCH_ENV := TORCHINDUCTOR_COMPILE_THREADS=`nproc`
# TORCH_ENV := TORCHINDUCTOR_COMPILE_THREADS=`nproc` TORCH_LOGS="output_code"
# TORCH_ENV := TORCHINDUCTOR_COMPILE_THREADS=`nproc` TORCH_LOGS="+dynamic"


flickr_8k_path := datasets/Flickr8k
flickr_8k_split := test

wmt14_de_en_path := datasets/wmt14/de-en
wmt14_de_en_split := test


RUN_RESNET152_SIM := $(TORCH_ENV) python -m example.resnet152 --data $(flickr_8k_path) --split $(flickr_8k_split) --device cuda
RUN_T5_SIM := $(TORCH_ENV) python -m example.t5 --data $(wmt14_de_en_path) --split $(wmt14_de_en_split) --device cuda


.PHONY: prepare_data
prepare_data: ## Download Flickr8k and WMT14 datasets
	$(PYTHON) scripts/download_datasets.py --flickr8k datasets/Flickr8k --wmt14 datasets/wmt14/de-en

run_resnet152_cycle: ## Run cycle simulation on ResNet-152 model
	$(RUN_RESNET152_SIM) --cycle --batch-size 4 --arch cigma
	$(RUN_RESNET152_SIM) --cycle --batch-size 4 --arch bitlet

run_t5_cycle: ## Run cycle simulation on T5 model
	$(RUN_T5_SIM) --cycle --batch-size 1 --arch cigma
	$(RUN_T5_SIM) --cycle --batch-size 1 --arch bitlet

run_t5_value: ## Run value simulation on T5 model
	$(RUN_T5_SIM) --value --batch-size 1 --arch cigma
	$(RUN_T5_SIM) --value --batch-size 1 --arch bitlet
