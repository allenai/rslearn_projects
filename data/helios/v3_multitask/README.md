## Finetuning on concatenated datasets

**Only one task per dataset is supported at the moment.**

There are two stages to running a multi-dataset job: constructing a run configuration, and actually running the job.

### 1. Constructing a run configuration

First write a multi-dataset config like `v3_multitask/small_v1.yaml`. This will be used to generate a run-specific config passed to `rslearn`.

Specify dataset-specific configs from `v2_*` in `task_cfgs`. If there are multiple config files, list them by override priority. Rather than specifying constants in a `launch_finetune` command, you must specify `patch_size`, `encoder_embedding_size`, and `helios_checkpoint_path` within the multi-dataset config itself. Be sure to also specify the `output_path` to the generated run configuration.

The `base_cfg` key points to `base.yaml` by default, which specifies the Helios encoder backbone structure,  training callbacks, etc. It can generally be kept as is, unless you need to modify the callbacks. 

Once you have constructed this multi-dataset config, you can generate the run config via

```bash
python make_multidataset_config.py --cfg [BASE_CONFIG]
```

### 2. Running a multi-dataset job

After generating a run config (see `v3_multitask/OUT*` for examples), it's straightforward to launch the multi-dataset job with `launch_finetune`. An example command is below. Note constants like `HELIOS_CHECKPOINT_PATH` are already substituted in by `make_multidataset_config`.

```bash
python -m rslp.main helios launch_finetune --config_paths+=[CONFIG] --rslp_project [PROJECT] --experiment_id [ID] --cluster+=[CLUSTER] --image_name [IMAGE_NAME]
```

Optionally, specify `--local true` to run in the current Beaker session. Please use the Beaker image `henryh/rslp_multidataset` for the latest version.