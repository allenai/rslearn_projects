###
### docker build --ssh default=/path/to/private.key -f Dockerfile .

ARG BASE=ubuntu:22.04
ARG BASE_PYTORCH=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80
ARG PLATFORM=linux/amd64

FROM --platform=${PLATFORM} ${BASE} AS tippecanoe

RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates curl libsqlite3-dev zlib1g-dev

RUN mkdir -p /tmp/tippecanoe && curl -L https://github.com/mapbox/tippecanoe/archive/refs/tags/1.36.0.tar.gz | tar -xz --strip 1 -C /tmp/tippecanoe
WORKDIR /tmp/tippecanoe
RUN make -j
RUN PREFIX=/opt/tippecanoe make install

# To use this:
#  COPY --from=tippecanoe /opt/tippecanoe /opt/tippecanoe
#  ENV PATH="/opt/tippecanoe/bin:${PATH}"

FROM --platform=${PLATFORM} ${BASE} AS golang

## Build Satlas smooth_point_labels_viterbi.go (Requires golang)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl
RUN curl -L https://go.dev/dl/go1.22.12.linux-amd64.tar.gz | tar -xz -C /usr/local
ENV PATH="/usr/local/go/bin:${PATH}"

COPY rslp/satlas/scripts /tmp/smooth_point_labels_viterbi/
WORKDIR /tmp/smooth_point_labels_viterbi/
RUN go build smooth_point_labels_viterbi.go

# To use this:
#  COPY --from=golang /tmp/smooth_point_labels_viterbi/smooth_point_labels_viterbi /usr/local/bin/smooth_point_labels_viterbi

FROM --platform=${PLATFORM} pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80 AS base

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git openssh-client

# Pin GitHub's host keys so SSH won't prompt
RUN mkdir -p -m 700 /root/.ssh && \
    ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /root/.ssh/known_hosts

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create virtual environment for package isolation
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

COPY . /opt/rslearn_projects/

# ============================================================================
# SSH Multi-Key Configuration for GitHub Actions
# ============================================================================
# Handles authentication for multiple private GitHub repos using separate deploy
# keys. See .github/workflows/build_test.yaml for full explanation of why this
# approach is needed and how it works. Local dev uses standard SSH agent forwarding.

ARG USE_SSH_KEYS_FROM_BUILD=false
RUN if [ "$USE_SSH_KEYS_FROM_BUILD" = "true" ] && [ -d /opt/rslearn_projects/.docker-ssh ]; then \
      echo "Setting up SSH keys from build context..." && \
      cp /opt/rslearn_projects/.docker-ssh/*_key /root/.ssh/ && \
      cp /opt/rslearn_projects/.docker-ssh/config /root/.ssh/config && \
      chmod 600 /root/.ssh/*_key /root/.ssh/config && \
      cp /opt/rslearn_projects/.docker-ssh/requirements-olmoearth_pretrain.txt /opt/rslearn_projects/requirements-olmoearth_pretrain.txt && \
      cp /opt/rslearn_projects/.docker-ssh/requirements-olmoearth_run.txt /opt/rslearn_projects/requirements-olmoearth_run.txt && \
      echo "SSH multi-key setup complete."; \
    else \
      echo "Using default SSH configuration (single key or SSH agent)."; \
    fi

RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install /opt/rslearn_projects[olmoearth_run,olmoearth_pretrain]

FROM --platform=${PLATFORM} ${BASE_PYTORCH} AS runner

# Copy only the virtual environment with installed packages (no source code, build tools, or dev dependencies)
COPY --from=base /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

FROM base AS full

COPY --from=tippecanoe /opt/tippecanoe /opt/tippecanoe
COPY --from=golang /tmp/smooth_point_labels_viterbi/smooth_point_labels_viterbi /usr/local/bin/smooth_point_labels_viterbi

ENV PATH="/opt/tippecanoe/bin:${PATH}"

## Install rslearn.
## We use git clone and then git checkout instead of git clone -b so that the user could
## specify a commit name or branch instead of only accepting a branch.
ARG RSLEARN_BRANCH=master
RUN --mount=type=ssh git clone git@github.com:allenai/rslearn.git /opt/rslearn
WORKDIR /opt/rslearn
RUN git checkout $RSLEARN_BRANCH
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /opt/rslearn[extra]

RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install /opt/rslearn_projects[dev,extra]
