## Ai2-Internal Documentation

## RSLP_PREFIX

rslearn_projects provides tooling on top of rslearn to launch training jobs on Beaker,
including code upload/download so that the exact code state is reproduced in the job.

Checkpoint management, W&B logging, and related model management are handled by
rslearn's built-in project management system (`--management_dir`).

Generally, we set `RSLP_PREFIX` on WEKA (via a `.env` file):

```
RSLP_PREFIX=/weka/dfive-default/rslearn-eai
```

Historically, we have also used GCS:

```
RSLP_PREFIX=gs://rslearn-eai
```

Model configuration files should include:

- `management_dir`: the directory for project management (typically `${RSLP_PREFIX}/projects`).
- `project_name`: a unique name for the project. This corresponds to the W&B project name.
- `run_name`: a unique name for this experiment. This corresponds to the W&B run name.

Checkpointing is handled by adding a `ManagedBestLastCheckpoint` callback to
`trainer.callbacks` in the model config:

```yaml
trainer:
  callbacks:
    - class_path: rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint
      init_args:
        monitor: val_loss
        mode: min
```

Files related to an experiment are stored in `{management_dir}/{project_name}/{run_name}/`.

There are command-line options provided by rslearn that control W&B logging and
checkpoint loading, see https://github.com/allenai/rslearn/blob/master/docs/ModelConfig.md#model-management-options.

## Beaker Sessions

To mount WEKA in Beaker sessions, launch sessions like this:

```
beaker session create --budget ai2/es-platform --workspace ai2/earth-systems --gpus 1 --shared-memory 256GiB --mount src=weka,ref=dfive-default,dst=/weka/dfive-default --bare --priority high
```

You can keep your code on WEKA or in `/data/[USERNAME]/`, the latter would only exist
within a single Beaker machine.

## Data Materialization and Training on Beaker

See `rslp/common/README.md` for information about launching data materialization jobs
and model training/prediction jobs on Beaker.
