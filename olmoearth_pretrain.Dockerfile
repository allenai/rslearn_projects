FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

RUN apt update
# g++ is needed at runtime by torch.compile to build CPU kernels.
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget g++

# Use uv to install everything.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies for rslearn, olmoearth_pretrain, olmoearth_run, and rslearn_projects.
COPY docker_build/rslearn/pyproject.toml /opt/rslearn/pyproject.toml
COPY docker_build/olmoearth_pretrain/pyproject.toml /opt/olmoearth_pretrain/pyproject.toml
COPY requirements.txt /opt/rslearn_projects/requirements.txt
COPY requirements-extra.txt /opt/rslearn_projects/requirements-extra.txt
COPY requirements-olmocore.txt /opt/rslearn_projects/requirements-olmocore.txt
COPY requirements-olmoearth_run.txt /opt/rslearn_projects/requirements-olmoearth_run.txt
# Using cache mount here avoids needing to re-download dependencies for later builds if the version didn't change.
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --break-system-packages /opt/rslearn[extra] /opt/olmoearth_pretrain -r /opt/rslearn_projects/requirements.txt -r /opt/rslearn_projects/requirements-extra.txt -r /opt/rslearn_projects/requirements-olmocore.txt -r /opt/rslearn_projects/requirements-olmoearth_run.txt

# Now copy the source code and install for real.
# If we don't change any dependencies, then only these steps need to be repeated
# (fast and means the new layers have small size).
COPY ./docker_build/rslearn /opt/rslearn
COPY ./docker_build/olmoearth_pretrain /opt/olmoearth_pretrain
COPY . /opt/rslearn_projects/

RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system --break-system-packages /opt/rslearn[extra] /opt/olmoearth_pretrain /opt/rslearn_projects[extra,olmoearth_run]

# Optional NATTEN wheel for Searchlight's attention kernel on H100, e.g.
# --build-arg NATTEN_VERSION=0.21.5+torch290cu128. It must match the installed torch.
ARG NATTEN_VERSION=
RUN if [ -n "$NATTEN_VERSION" ]; then \
      uv pip install --system --break-system-packages \
        --find-links https://whl.natten.org "natten==$NATTEN_VERSION"; \
    fi

WORKDIR /opt/rslearn_projects

# Recorded in large_scale_embeddings completion markers and provenance.
ARG CODE_VERSION=unknown
ENV RSLP_CODE_VERSION=${CODE_VERSION}
