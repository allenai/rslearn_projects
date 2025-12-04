## Ai2-Internal Documentation

## RSLP_PREFIX

rslearn_projects provides tooling on top of rslearn to automatically determine where to
store model checkpoints and related files within the directory specified by the
`RSLP_PREFIX` environment variable.

Generally, we set `RSLP_PREFIX` on WEKA (via a `.env` file):

```
RSLP_PREFIX=/weka/dfive-default/rslearn-eai
```

Historically, we have also used GCS:

```
RSLP_PREFIX=gs://rslearn-eai
```

Model configuration files for rslearn_projects have two additional fields:

- `rslp_project`: a unique name for the project for which this is one experiment. This
  corresponds to the W&B project name.
- `rslp_experiment`: a unique name for this experiment. This corresponds to the W&B run
  name.

Note: similar project management functionality has since been added to rslearn, and we
should move to just using that eventually, but for now we still use the system in
rslearn_projects which uses slightly different paths and config options.

Then, files related to that experiment would be stored in
`{RSLP_PREFIX}/projects/{rslp_project}/{rslp_experiment}`.

There are a few command-line options provided by rslearn_projects that control W&B
logging and checkpoint loading (see `rslp/lightning_cli.py` for all options):

- `--no_log=true`: by default, a W&B logger is automatically configured during
  `model fit` even if it doesn't appear in the model config file (under `trainer:`
  section). This option will prevent adding the logger.
- `--force_log=true`: by default, W&B logging is not enabled for test/predict. This
  option will add the logger even for these other subcommands.
- `--autoresume=true`: by default, the system will raise an error if there is an
  existing checkpoint. This option will load the latest checkpoint (`last.ckpt`)
  instead, if it exists. It is typically enabled during training.
- `--load_best=true`: like autoresume but instead of loading the latest checkpoint, it
  will load the best checkpoint (typically named like `epoch=X....ckpt`). Unlike
  autoresume, it will raise an error if there is no existing checkpoint.

These are all passed through the `rslp.rslearn_main` entrypoint, e.g.:

```
python -m rslp.rslearn_main model fit --config data/20251104/config.yaml` --autoresume=true
```

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
