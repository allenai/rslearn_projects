ARG PYTORCH_BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

FROM ${PYTORCH_BASE_IMAGE} AS base

RUN apt-get update &&  \
    apt-get install --no-install-recommends -y libpq5 ffmpeg libsm6 libxext6 git wget curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure uv for Docker usage
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV ESRUN_API_URL=https://fake-api-host:8000

# Copy project files for dependency resolution
COPY ./repos/helios /opt/helios
#COPY ./repos/rslearn /opt/rslearn
COPY ./repos/earth-system-run /opt/esrun
COPY requirements.txt /opt/rslearn_projects/
COPY pyproject_esrunner.toml /opt/rslearn_projects/pyproject.toml

WORKDIR /opt/rslearn_projects

# Install all dependencies including local packages with uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --extra local

ENV PATH=$PATH:/opt/rslearn_projects/.venv/bin

CMD ["esrunner"]
