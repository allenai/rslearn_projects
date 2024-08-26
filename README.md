Overview
--------

rslearn_projects contains Ai2-specific tooling for managing remote sensing projects
built on top of rslearn, as well as project-specific code and configuration files.


Tooling
-------

The additional tooling comes into play when training and deploying models. This is an
outline of the steps the tooling takes care of when training models:

1. User runs e.g. `python -m rslp.launch_beaker --config_path path/to/config.yaml`.
2. Launcher uploads the code to a canonical path on Google Cloud Storage (GCS), based
   on the project ID and experiment ID specified in `config.yaml`.
3. Launcher then starts a job, in this case on Beaker, to train the model.
4. `rslp.docker_entrypoint` is the entrypoint for the job, and starts by downloading
   the code. The image contains a copy of the code too, but it is overwritten with the
   latest code from the user's codebase.
5. It then saves W&B run ID to GCS. It also configures rslearn to write checkpoints to
   a canonical folder on GCS.
6. If the job is pre-empted and resumes, it will automatically load the latest
   checkpoint and W&B run ID from GCS. It will also load these in calls to `model test`
   or `model predict`.


Usage
-----

Credentials for GCP, Weka, and W&B are stored in secrets so you don't need to worry
about setting them up. But you do need to setup your Beaker or other platform
credentials to be able to launch the jobs.

TODO: update GCP/W&B to use service accounts.

Create an environment for rslearn and setup with rslearn_projects requirements:

    conda create -n rslearn python=3.12
    conda activate rslearn
    pip install -r rslearn/requirements.txt
    pip install -r rslearn_projects/requirements.txt

Then run the launcher:

    python -m rslp.launch_beaker --config_path maldives_ecosystem_mapping/train/config.yaml

Or manually run training:

    python -m rslp.rslearn_main model fit --config_path maldives_ecosystem_mapping/train/config.yaml
