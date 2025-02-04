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

    python -m rslp.main common worker skylight-proto-1 rslp-job-queue-YOURNAME-sub

And to launch 100 workers on Beaker:

    python -m rslp.main common launch BEAKER_IMAGE_NAME skylight-proto-1 rslp-job-queue-YOURNAME-sub 100 --gpus 1 --shared_memory 256GiB


Beaker Launcher
---------------

`beaker_launcher.py` launches a Beaker job that runs an rslp workflow. It offers a
range of parameters to customize the job setup, such as which Beaker clusters to target
and application-specific environment variables.
