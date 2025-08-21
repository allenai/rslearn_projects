FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget

# Install uv - there is an issue with lightning versioning, so hack is to use uv pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Prep for rslearn and helios install (need to be in local directory).
COPY ./docker_build/rslearn /opt/rslearn
COPY ./docker_build/helios /opt/helios
COPY requirements.txt /opt/rslearn_projects/requirements.txt
COPY ai2_requirements.txt /opt/rslearn_projects/ai2_requirements.txt

# We also install terratorch so that we can use the same Docker image for TerraMind
# experiments, as well as geobench.
RUN uv pip install --system --no-cache-dir git+https://github.com/IBM/terratorch.git
RUN uv pip install --system --no-cache-dir geobench==0.0.1

# Install rslearn and helios.
RUN uv pip install --system --no-cache-dir --upgrade /opt/rslearn[extra]
RUN uv pip install --system --no-cache-dir --upgrade /opt/helios
RUN uv pip install --system --no-cache-dir -r /opt/rslearn_projects/requirements.txt -r /opt/rslearn_projects/ai2_requirements.txt

# Copy rslearn_projects and install it too.
COPY . /opt/rslearn_projects/
RUN uv pip install --system --no-cache-dir /opt/rslearn_projects

WORKDIR /opt/rslearn_projects
