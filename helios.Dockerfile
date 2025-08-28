FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget

# Install rslearn and helios (need to be in local directory).
COPY ./docker_build/rslearn /opt/rslearn
COPY ./docker_build/helios /opt/helios

# We also install terratorch so that we can use the same Docker image for TerraMind
# experiments.
RUN pip install --no-cache-dir git+https://github.com/IBM/terratorch.git
RUN pip install --no-cache-dir geobench==0.0.1

RUN pip install --no-cache-dir --upgrade /opt/rslearn[extra]
RUN pip install --no-cache-dir --upgrade /opt/helios

COPY requirements-without-rslearn.txt /opt/rslearn_projects/requirements-without-rslearn.txt
COPY requirements-extra.txt /opt/rslearn_projects/requirements-extra.txt
RUN pip install --no-cache-dir -r /opt/rslearn_projects/requirements-without-rslearn.txt -r /opt/rslearn_projects/requirements-extra.txt

# Copy rslearn_projects and install it too.
COPY . /opt/rslearn_projects/
RUN pip install --no-cache-dir /opt/rslearn_projects

WORKDIR /opt/rslearn_projects
