# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

# Cigma-Sim develop tool targets


ALL_PYTHON_DIR := cigmasim example

RUFF_FORMAT_UNSAFE_FIX_RULES := ANN201,ANN204,B007,D212,D400,D403,I001,PERF102,SIM118,TID252,UP007

RUFF_LINT_EXTEND_IGNORE := # --extend-ignore


.PHONY: format
format: ## Run `ruff` formatter with auto fix
	ruff format $(ALL_PYTHON_DIR)
	ruff check --fix-only $(ALL_PYTHON_DIR) --unsafe-fixes --select $(RUFF_FORMAT_UNSAFE_FIX_RULES)
	ruff check --fix-only $(ALL_PYTHON_DIR)

.PHONY: lint
lint: format ## Run `ruff` linter
	ruff check $(ALL_PYTHON_DIR) $(RUFF_LINT_EXTEND_IGNORE) 2>&1 | tee ruff_report.log || true

.PHONY: check
check: ## Run `mypy` static analysis
	mypy $(ALL_PYTHON_DIR) 2>&1 | tee mypy_report.log || true

