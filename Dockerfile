FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git
COPY rslearn/requirements.txt /opt/rslearn_projects/rslearn/requirements.txt
RUN pip install -r /opt/rslearn_projects/rslearn/requirements.txt
COPY rslearn/extra_requirements.txt /opt/rslearn_projects/rslearn/extra_requirements.txt
RUN pip install -r /opt/rslearn_projects/rslearn/extra_requirements.txt
COPY requirements.txt /opt/rslearn_projects/requirements.txt
RUN pip install -r /opt/rslearn_projects/requirements.txt

# Clone and install SAM2 (Segment Anything 2)
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /opt/segment-anything-2
RUN pip install -e /opt/segment-anything-2

ENV PYTHONPATH="${PYTHONPATH}:/opt/rslearn_projects/rslearn:."

COPY /. /opt/rslearn_projects/
WORKDIR /opt/rslearn_projects
