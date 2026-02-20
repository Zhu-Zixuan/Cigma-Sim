# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

# Cigma-Sim container tool targets


CONTAINER_ENGINE ?= docker
# CONTAINER_ENGINE ?= podman


CONTAINER_PYTORCH_IMAGE_NAME := docker.io/pytorch/pytorch
CONTAINER_PYTORCH_IMAGE_TAG := 2.6.0-cuda12.6-cudnn9-devel

CONTAINER_PYTORCH_IMAGE := $(CONTAINER_PYTORCH_IMAGE_NAME):$(CONTAINER_PYTORCH_IMAGE_TAG)
CONTAINER_PYTORCH_IMAGE_VERSION := $(word 1,$(subst -, ,$(CONTAINER_PYTORCH_IMAGE_TAG)))


CONTAINER_DOCKERFILE := $(PROJ_ROOT)/container/Dockerfile


CONTAINER_IMAGE_NAME := $(PROJ_NAME)

CONTAINER_IMAGE_TAG_RUNTIME := pytorch$(CONTAINER_PYTORCH_IMAGE_VERSION)-runtime
CONTAINER_IMAGE_TAG_DEVELOP := pytorch$(CONTAINER_PYTORCH_IMAGE_VERSION)-devel

CONTAINER_IMAGE_RUNTIME := $(CONTAINER_IMAGE_NAME):$(CONTAINER_IMAGE_TAG_RUNTIME)
CONTAINER_IMAGE_DEVELOP := $(CONTAINER_IMAGE_NAME):$(CONTAINER_IMAGE_TAG_DEVELOP)


# --- Check Build Dependencies ---

.PHONY: container_check_build_deps
container_check_build_deps: container_check_pyproject_toml # Check build dependencies

.PHONY: container_check_pyproject_toml
container_check_pyproject_toml: # Check if `pyproject.toml` exists in project root
	@if [ ! -f "$(PROJ_ROOT)/pyproject.toml" ]; then \
		echo "Error: pyproject.toml not found in project root $(PROJ_ROOT)."; \
		exit 1; \
	fi


# --- Runtime Container ---

.PHONY: container_build_runtime
container_build_runtime: container_check_build_deps ## Build runtime container image
	$(CONTAINER_ENGINE) build -f "$(CONTAINER_DOCKERFILE)" --target runtime \
		--build-arg "BASE_IMAGE=$(CONTAINER_PYTORCH_IMAGE)" \
		-t "$(CONTAINER_IMAGE_RUNTIME)" \
		"$(PROJ_ROOT)"


# --- Develop Container ---

.PHONY: container_build_devel
container_build_devel: container_check_build_deps ## Build dev container image
	$(CONTAINER_ENGINE) build -f "$(CONTAINER_DOCKERFILE)" --target devel \
		--build-arg "BASE_IMAGE=$(CONTAINER_PYTORCH_IMAGE)" \
		-t "$(CONTAINER_IMAGE_DEVELOP)" \
		"$(PROJ_ROOT)"
