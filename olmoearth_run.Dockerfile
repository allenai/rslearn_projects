FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install rslearn.
# We use git clone and then git checkout instead of git clone -b so that the user could
# specify a commit name or branch instead of only accepting a branch.
ARG RSLEARN_BRANCH=master
RUN git clone https://github.com/allenai/rslearn.git /opt/rslearn
WORKDIR /opt/rslearn
RUN git checkout $RSLEARN_BRANCH
RUN uv pip install --system /opt/rslearn[extra]

# Install rslearn_projects.
COPY . /opt/rslearn_projects/
RUN uv pip install --system /opt/rslearn_projects[dev,extra]

# Install olmoearth_pretrain and olmoearth_run dependencies.
COPY requirements-olmoearth_pretrain.txt requirements-olmoearth_run.txt /opt/rslearn_projects/
RUN --mount=type=secret,id=github_token \
    git config --global url."https://$(cat /run/secrets/github_token)@github.com/".insteadOf "https://github.com/" && \
    uv pip install --system -r /opt/rslearn_projects/requirements-olmoearth_pretrain.txt && \
    uv pip install --system -r /opt/rslearn_projects/requirements-olmoearth_run.txt

WORKDIR /opt/rslearn_projects
