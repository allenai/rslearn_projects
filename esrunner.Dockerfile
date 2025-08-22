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

# This is just a placeholder and will need to be replaced with the actual API URL at runtime.
ENV ESRUN_API_URL=https://fake-api-host:8000

# Copy helios, esrun, and requirements files into the image
COPY ./repos/helios /opt/helios
COPY ./repos/earth-system-run /opt/esrun
COPY requirements.txt /opt/rslearn_projects/
COPY pyproject_esrunner.toml /opt/rslearn_projects/pyproject.toml

WORKDIR /opt/rslearn_projects

# Create the virtual environment, keeping system packages in scope, and then install all dependencies
#  including local packages with uv uses the pyproject.esrunner.toml that gets copied into the image.
RUN uv venv --system-site-packages
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --extra local

ENV PATH=$PATH:/opt/rslearn_projects/.venv/bin

CMD ["esrunner"]
