FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime@sha256:58a28ab734f23561aa146fbaf777fb319a953ca1e188832863ed57d510c9f197

# TEMPORARY Until RSLEARN Is Public
ARG GIT_USERNAME
ARG GIT_TOKEN

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git
RUN git clone https://${GIT_USERNAME}:${GIT_TOKEN}@github.com/allenai/rslearn.git /opt/rslearn_projects/rslearn
RUN pip install -r /opt/rslearn_projects/rslearn/requirements.txt
RUN pip install -r /opt/rslearn_projects/rslearn/extra_requirements.txt
COPY requirements.txt /opt/rslearn_projects/requirements.txt
RUN pip install -r /opt/rslearn_projects/requirements.txt

# We need rslp to be pip installed as well

ENV PYTHONPATH="${PYTHONPATH}:/opt/rslearn_projects/rslearn:."

COPY /. /opt/rslearn_projects/
WORKDIR /opt/rslearn_projects
