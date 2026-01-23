FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget

# Use uv to install everything.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies for rslearn, olmoearth_pretrain, and rslearn_projects.
COPY docker_build/rslearn/pyproject.toml /opt/rslearn/pyproject.toml
COPY docker_build/olmoearth_pretrain/pyproject.toml /opt/olmoearth_pretrain/pyproject.toml
COPY requirements.txt /opt/rslearn_projects/requirements.txt
COPY requirements-extra.txt /opt/rslearn_projects/requirements-extra.txt
# Using cache mount here avoids needing to re-download dependencies for later builds if the version didn't change.
RUN uv pip install --system /opt/rslearn[extra] /opt/olmoearth_pretrain -r /opt/rslearn_projects/requirements.txt -r /opt/rslearn_projects/requirements-extra.txt

# Now copy the source code and install for real.
# If we don't change any dependencies, then only these steps need to be repeated
# (fast and means the new layers have small size).
COPY ./docker_build/rslearn /opt/rslearn
COPY ./docker_build/olmoearth_pretrain /opt/olmoearth_pretrain
COPY . /opt/rslearn_projects/

RUN uv pip install --system /opt/rslearn[extra] /opt/olmoearth_pretrain /opt/rslearn_projects[extra]

WORKDIR /opt/rslearn_projects
