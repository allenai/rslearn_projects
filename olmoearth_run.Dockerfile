# olmoearth-runner requires Python 3.12, which the 2.7.0 image does not have.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget

# Install rslearn and olmoearth_pretrain (need to be in local directory).
COPY ./docker_build/rslearn /opt/rslearn
COPY ./docker_build/olmoearth_pretrain /opt/olmoearth_pretrain

RUN pip install --no-cache-dir --upgrade /opt/rslearn[extra]
RUN pip install --no-cache-dir --upgrade /opt/olmoearth_pretrain

COPY requirements-without-rslearn.txt /opt/rslearn_projects/requirements-without-rslearn.txt
COPY requirements-extra.txt /opt/rslearn_projects/requirements-extra.txt
COPY requirements-olmoearth_run.txt /opt/rslearn_projects/requirements-olmoearth_run.txt
RUN pip install --no-cache-dir -r /opt/rslearn_projects/requirements-without-rslearn.txt -r /opt/rslearn_projects/requirements-extra.txt -r /opt/rslearn_projects/requirements-olmoearth_run.txt

# Copy rslearn_projects and install it too.
COPY . /opt/rslearn_projects/
RUN pip install --no-cache-dir /opt/rslearn_projects

WORKDIR /opt/rslearn_projects
