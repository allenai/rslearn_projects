Overview
--------

rslearn_projects contains Ai2-specific tooling for managing remote sensing projects
built on top of rslearn, as well as project-specific code and configuration files.


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
launcher like `rslp.main common beaker_train`, then it isn't needed since the credentials are
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

Manually train locally:

    python -m rslp.rslearn_main model fit --config_path data/maldives_ecosystem_mapping/config.yaml

To launch training on Beaker, first build and push a Docker image:

    docker build -t rslp .
    beaker image create --name rslp rslp

Launch training on Beaker:

    python -m rslp.main common beaker_train --config_path data/maldives_ecosystem_mapping/config_planetscope_plus_sentinel2.yaml --image_name [YOUR_BEAKER_USERNAME]/rslp --cluster '["ai2/jupiter-cirrascale-2"]'

Note that `rslp.main` is the entrypoint for rslp pipelines (see Pipelines section
below) while `rslp.rslearn_main` simply wraps the rslearn commands but with the various
functionality in `rslearn_projects` added in.


Tooling for Model Training and Inference
----------------------------------------

The additional tooling comes into play when training and deploying models. This is an
outline of the steps the tooling takes care of when training models:

1. User runs e.g. `python -m rslp.main common beaker_train --config_path path/to/config.yaml`.
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


Pipelines
---------

There are several "pipelines" (aka "workflows") implemented in rslearn_projects that
can be called from the command-line with arguments parsed via jsonargparse.

For example, the Satlas prediction pipeline is called like this (see
`rslp/satlas/README.md` for details):

    python -m rslp.main satlas predict --application SOLAR_FARM ...

A few pipelines, including the Satlas prediction pipeline, are designed to be run at
scale in Beaker jobs. These typically use the worker system implemented in
`rslp.common.worker` where tasks that involve executing a rslp pipeline with certain
arguments can be written to a Beaker queue; then, Beaker jobs can be launched that
process tasks from that queue.


Beaker Secrets
--------------

We have set up these Beaker secrets for all Beaker jobs:

- RSLEARN_WANDB_API_KEY: the W&B API key.
- RSLEARN_WEKA_KEY: access key ID for WEKA (when using S3-compatible API).
- RSLEARN_WEKA_SECRET: the secret access key for WEKA (when using S3-compatible API).
- RSLP_BEAKER_TOKEN: the Beaker token.
- RSLEARN_GCP_CREDENTIALS: GCP credentials, mounted at `/etc/credentials/gcp_credentials.json`.

The RSLP_PREFIX will be set from the user's environment, which should be either
`gs://rslearn-eai/` or `/weka/dfive-default/rslearn-eai/`.

Certain job launchers may also:

- Several launchers will mount the `dfive-default` WEKA bucket to
  `/weka/dfive-default/`.
- `common.beaker_data_materialization` will mount GCP_HELIOS_SERVICE_ACCOUNT to
  `/etc/credentials/gee_credentials.json` (with credentials for Google Earth Engine
  service account).

These environment variables are also set:

- GCLOUD_PROJECT and GOOGLE_CLOUD_PROJECT: earthsystem-dev-c3po
- WEKA_ENDPOINT_URL: https://weka-aus.beaker.org:9000
- GOOGLE_APPLICATION_CREDENTIALS: "/etc/credentials/gcp_credentials.json" (where the
  RSLEARN_GCP_CREDENTIALS secret is mounted).


Miscellaneous Tooling
---------------------

There are several other components in rslearn_projects, so here is a list of some of
them.

- Helios model wrapper: see `rslp/helios/README.md` for usage.
- `rslp/common/README.md` documents the worker system (to launch Beaker jobs that
  process tasks from a Beaker queue, where the tasks involve running a "pipeline" in
  rslp), the `beaker_train` command, and the `launch_data_materialization_jobs`
  command.
