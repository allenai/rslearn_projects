FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime@sha256:7db0e1bf4b1ac274ea09cf6358ab516f8a5c7d3d0e02311bed445f7e236a5d80

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget

# Install Go (used for Satlas smooth_point_labels_viterbi.go).
RUN wget https://go.dev/dl/go1.22.12.linux-amd64.tar.gz
RUN rm -rf /usr/local/go && tar -C /usr/local -xzf go1.22.12.linux-amd64.tar.gz
ENV PATH="${PATH}:/usr/local/go/bin"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install rslearn_projects (rslearn is pinned to a specific commit in requirements.txt).
COPY . /opt/rslearn_projects/
RUN uv pip install --system /opt/rslearn_projects[dev,extra,olmoearth_pretrain]

# Build Satlas smooth_point_labels_viterbi.go program.
WORKDIR /opt/rslearn_projects/rslp/satlas/scripts
RUN go build smooth_point_labels_viterbi.go

WORKDIR /opt/rslearn_projects
