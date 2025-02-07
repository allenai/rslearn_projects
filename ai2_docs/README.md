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


Setup
-----

rslp expects an environment variable specifying the GCS bucket to write prepared
rslearn datasets, model checkpoints, etc. The easiest way is to create a `.env` file.

    RSLP_PREFIX=gs://rslearn-eai
    RSLP_WEKA_PREFIX=weka://dfive-default/rslearn-eai
    BKT_PROJECT_ID=ai2-prior-satlas
    BKT_BUCKET_NAME=satlas-explorer-data
    BKT_BIGTABLE_PROJECT_ID=ai2-prior-satlas
    BKT_BIGTABLE_INSTANCE_ID=satlas

You will also need to setup GCP credentials that have access to this bucket.

Training additionally depends on credentials for W&B. If you train directly using
`rslp.rslearn_main`, then you will need to setup these credentials. If you use a
launcher like `rslp.launch_beaker`, then it isn't needed since the credentials are
already configured as secrets on the platform, but you would need to setup your Beaker
or other platform credentials to be able to launch the jobs.


Usage
-----

Create an environment for rslearn and setup with rslearn_projects requirements:

    conda create -n rslearn python=3.12
    conda activate rslearn
    pip install -r rslearn/requirements.txt -r rslearn/extra_requirements.txt
    pip install -r rslearn_projects/requirements.txt

For development it is easier to use PYTHONPATH or install rslearn and rslearn_projects
in editable mode, e.g.:

    export PYTHONPATH=.:/path/to/rslearn/rslearn

Execute a data processing pipeline:

    python -m rslp.main maldives_ecosystem_mapping data --dp_config.workers 32

Launch training on Beaker:

    python -m rslp.main maldives_ecosystem_mapping train_maxar

Manually train locally:

    python -m rslp.rslearn_main model fit --config_path data/maldives_ecosystem_mapping/config.yaml


Projects
--------

- [Forest Loss Driver](rslp/forest_loss_driver/README.md)
