# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

# Cigma-Sim Makefile


PROJ_ROOT := $(shell pwd)
PROJ_NAME := cigma-sim


include container/container.mk


# --- Develop ---

ALL_PYTHON_DIR := cigmasim example

.PHONY: format
format: ## Run `ruff` formatter with auto fix
	ruff format $(ALL_PYTHON_DIR)
	ruff check --fix-only $(ALL_PYTHON_DIR)

.PHONY: lint
lint: format ## Run `ruff` linter
	ruff check $(ALL_PYTHON_DIR) 2>&1 | tee ruff_report.log

.PHONY: check
check: ## Run `mypy` static analysis
	mypy $(ALL_PYTHON_DIR) 2>&1 | tee mypy_report.log


# --- Cleaning ---

.PHONY: clean
clean: ## Clean up cache and temporary files
	rm -rf .*_cache *.log site .coverage htmlcov log
	rm -rf ~/.triton/cache /tmp/torchinductor_*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +


# --- Help Info ---

.DEFAULT_GOAL := help
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
