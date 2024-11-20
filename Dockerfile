FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime@sha256:d15e9803095e462e351f097fb1f5e7cdaa4f5e855d7ff6d6f36ec4c2aa2938ea

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git

# Install rslearn.
# We use git clone and then git checkout instead of git clone -b so that the user could
# specify a commit name or branch instead of only accepting a branch.
ARG RSLEARN_BRANCH=master
RUN git clone https://github.com/allenai/rslearn.git /opt/rslearn
WORKDIR /opt/rslearn
RUN git checkout $RSLEARN_BRANCH
RUN pip install --no-cache-dir /opt/rslearn[extra]

# maybe some steps to make this huge iamge smaller

# Install rslearn_projects dependencies.
# We do this in a separate step so it doesn't need to be rerun when other parts of the
# context are modified.
COPY requirements.txt /opt/rslearn_projects/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /opt/rslearn_projects/requirements.txt

# Copy rslearn_projects.
# For now we don't install it and instead just use PYTHONPATH.
ENV PYTHONPATH="${PYTHONPATH}:."

COPY /. /opt/rslearn_projects/
# install rslp package
RUN pip install --no-cache-dir /opt/rslearn_projects
WORKDIR /opt/rslearn_projects
