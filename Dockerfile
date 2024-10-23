FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime@sha256:58a28ab734f23561aa146fbaf777fb319a953ca1e188832863ed57d510c9f197

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git

# Install rslearn.
ARG RSLEARN_BRANCH=master
RUN git clone -b $RSLEARN_BRANCH https://github.com/allenai/rslearn.git
RUN cd ./rslearn
RUN pip install --no-cache-dir --upgrade --upgrade-strategy eager .[extra]

# Install rslearn_projects dependencies.
# We do this in a separate step so it doesn't need to be rerun when other parts of the
# context are modified.
COPY requirements.txt /opt/rslearn_projects/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /opt/rslearn_projects/requirements.txt

# Copy rslearn_projects.
# For now we don't install it and instead just use PYTHONPATH.
ENV PYTHONPATH="${PYTHONPATH}:."
COPY /. /opt/rslearn_projects/
WORKDIR /opt/rslearn_projects
