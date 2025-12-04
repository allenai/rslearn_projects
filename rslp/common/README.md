This contains infrastructure intended to be shared across several rslp projects.


Worker
------

`worker.py` provides a system for launching Beaker jobs that run workers to execute
tasks from a queue.

Each task specifies an rslp project, pipeline (workflow), and arguments to pass to the
pipeline. The queue is implemented with Google Cloud Pub/Sub.

First create the topic and subscription via CLI:

    gcloud pubsub topics create --project skylight-proto-1 rslp-job-queue-YOURNAME
    gcloud pubsub subscriptions create --project skylight-proto-1 rslp-job-queue-YOURNAME-sub --topic rslp-job-queue-YOURNAME

You will then need to write code that writes tasks to the topic.
See `satlas/write_jobs.py` for an example of this.

Then you can launch the worker. To test on one machine:

    python -m rslp.main common worker --project_id skylight-proto-1 --subscription_id rslp-job-queue-YOURNAME-sub

And to launch 100 workers on Beaker:

    python -m rslp.main common launch --image_name BEAKER_IMAGE_NAME --project_id skylight-proto-1 --subscription_id rslp-job-queue-YOURNAME-sub --num_workers 100 --gpus 1 --shared_memory 256GiB --cluster '["ai2/jupiter-cirrascale-2"]'


Beaker Launcher
---------------

`beaker_launcher.py` launches a Beaker job that runs an rslp workflow. It offers a
range of parameters to customize the job setup, such as which Beaker clusters to target
and application-specific environment variables.


Beaker Data Materialization
---------------------------

`beaker_data_materialization.py` launches Beaker jobs for materializing data in rslearn
datasets. The command to run can be overridden via an argument to prepare or ingest
data instead.

First, build a Docker image with rslearn and rslearn_projects using the Dockerfile (or
olmoearth_pretrain.Dockerfile), and push it as a Beaker image.

Then, launch the Beaker jobs. The hostnames should be specified to ensure that the CPU
Beaker jobs are scheduled on different hosts. WEKA is mounted so the `--ds_path` could
be on WEKA instead.

    python -m rslp.main common launch_data_materialization_jobs --image favyen/rslp_image --ds_path gs://path/to/rslearn_dataset/ --hosts+=jupiter-cs-aus-134.reviz.ai2.in

Here is an example overriding the command to ingest instead:

    python -m rslp.main common launch_data_materialization_jobs --image favyen/rslp_image --ds_path gs://path/to/rslearn_dataset/ --hosts+=jupiter-cs-aus-134.reviz.ai2.in --command '["rslearn", "dataset", "ingest", "--root", "gs://path/to/rslearn_dataset/", "--workers", "64", "--no-use-initial-job", "--retry-max-attempts", "5", "--retry-backoff-seconds", "60", "--ignore-errors"]'


Beaker Training and Prediction
------------------------------

In rslearn_projects, model commands are run through the `rslp.rslearn_main` endpoint to
take advantage of specialized checkpoint and logging handling.

We can also run that in Beaker jobs instead of locally. First, build a Docker image
using `olmoearth_pretrain.Dockerfile` and push it to Beaker, per the instructions at
`rslp/olmoearth_pretrain/README.md`.

To launch a training job, e.g.:

```
python -m rslp.main common beaker_train --image_name YOUR_BEAKER_IMAGE --cluster+=ai2/jupiter --gpus 1 --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' --priority high
```

To launch a prediction job, e.g.:

```
python -m rslp.main common beaker_train --image_name YOUR_BEAKER_IMAGE --cluster+=ai2/jupiter --gpus 1 --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' --priority high --mode predict --extra_args='["--data.init_args.path", "/path/to/dataset"]'
```
