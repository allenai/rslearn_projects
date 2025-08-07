ARG PYTORCH_BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

FROM ${PYTORCH_BASE_IMAGE} AS base

RUN apt-get update &&  \
    apt-get install --no-install-recommends -y libpq5 ffmpeg libsm6 libxext6 git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/root/.local/bin

# Install rslearn and helios (need to be in local directory).
COPY ./repos/helios /opt/helios
#COPY ./repos/rslearn /opt/rslearn
COPY ./repos/earth-system-run /opt/esrun
COPY requirements.txt /opt/rslearn_projects/

RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install --user /opt/helios /opt/esrun[runner]
RUN --mount=type=cache,target=/root/.cache/pip pip install --user -r /opt/rslearn_projects/requirements.txt

WORKDIR /opt/rslearn_projects
