FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6
COPY rslearn/requirements.txt /opt/rslearn_projects/rslearn/requirements.txt
RUN pip install -r /opt/rslearn_projects/rslearn/requirements.txt
COPY rslearn/extra_requirements.txt /opt/rslearn_projects/rslearn/extra_requirements.txt
RUN pip install -r /opt/rslearn_projects/rslearn/extra_requirements.txt
COPY requirements.txt /opt/rslearn_projects/requirements.txt
RUN pip install -r /opt/rslearn_projects/requirements.txt
ENV PYTHONPATH="${PYTHONPATH}:/opt/rslearn_projects/rslearn:."

COPY /. /opt/rslearn_projects/
WORKDIR /opt/rslearn_projects
