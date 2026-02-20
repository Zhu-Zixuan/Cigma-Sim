# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT


ARG BASE_IMAGE


# === runtime image ===

FROM ${BASE_IMAGE} AS runtime

# pytorch base image provide `conda`, use it
RUN conda install -y --no-update-deps \
    --no-channel-priority --override-channels -c conda-forge \
    make uv \
    && conda clean -afy

COPY pyproject.toml /tmp/pyproject.toml

# extract dependencies from `pyproject.toml` as the single source of truth
RUN uv pip install --quiet --no-cache --system -r /tmp/pyproject.toml \
    && uv cache clean \
    && rm -rf /tmp/pyproject.toml

CMD ["/bin/bash"]



# === devel image ===

FROM ${BASE_IMAGE} AS devel

# pytorch base image provide `conda`, use it
RUN conda install -y --no-update-deps \
    --no-channel-priority --override-channels -c conda-forge \
    make uv git nvitop \
    && conda clean -afy

COPY pyproject.toml /tmp/pyproject.toml

# extract dependencies from `pyproject.toml` as the single source of truth
RUN uv pip install --quiet --no-cache --system -r /tmp/pyproject.toml \
    --extra dev \
    && uv cache clean \
    && rm -rf /tmp/pyproject.toml

CMD ["/bin/bash"]
