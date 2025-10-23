FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget

# Install rslearn, olmoearth_pretrain and olmoearth_run (need to be in local directory).
COPY ./docker_build/rslearn /opt/rslearn
COPY ./docker_build/olmoearth_pretrain /opt/olmoearth_pretrain
COPY ./docker_build/olmoearth_run /opt/olmoearth_run

RUN pip install --no-cache-dir --upgrade /opt/rslearn[extra]
RUN pip install --no-cache-dir --upgrade /opt/olmoearth_pretrain
RUN pip install --no-cache-dir --upgrade /opt/olmoearth_run

COPY requirements-without-rslearn.txt /opt/rslearn_projects/requirements-without-rslearn.txt
COPY requirements-extra.txt /opt/rslearn_projects/requirements-extra.txt
RUN pip install --no-cache-dir -r /opt/rslearn_projects/requirements-without-rslearn.txt -r /opt/rslearn_projects/requirements-extra.txt

# Copy rslearn_projects and install it too.
COPY . /opt/rslearn_projects/
RUN pip install --no-cache-dir /opt/rslearn_projects

WORKDIR /opt/rslearn_projects
