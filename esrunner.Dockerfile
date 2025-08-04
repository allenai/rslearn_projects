FROM us-west1-docker.pkg.dev/earthsystem-shared/earthsystem-docker-images/vendor/nvidia/pytorch:25.06-py3 AS os_base
#FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
#FROM nvcr.io/nvidia/pytorch:25.06-py3
# FROM python:3.12 AS os_base

RUN apt-get update &&  \
    apt-get install -y libpq5 ffmpeg libsm6 libxext6 git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM python:3.12 AS setup_python

ENV PATH=$PATH:/root/.local/bin

# Install rslearn and helios (need to be in local directory).
COPY ./repos/helios /opt/helios
COPY ./repos/rslearn /opt/rslearn
COPY ./repos/earth-system-run /opt/esrun
COPY requirements.txt /opt/rslearn_projects/

RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install --user /opt/rslearn[extra] /opt/helios /opt/esrun[runner]
RUN --mount=type=cache,target=/root/.cache/pip pip install --user -r /opt/rslearn_projects/requirements.txt
#RUN pip install --user --no-cache-dir /opt/esrun[runner]
#COPY . /opt/rslearn_projects/
#
#RUN pip install --upgrade pip
#RUN pip install --no-cache-dir /opt/rslearn[extra]
#RUN pip install /opt/helios
#RUN pip install --no-cache-dir /opt/rslearn_projects

WORKDIR /opt/rslearn_projects
