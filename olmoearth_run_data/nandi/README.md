Local olmoearth_run for Nandi:
```bash
export EXTRA_FILES_PATH=/weka/dfive-default/helios/checkpoints
export DATASET_PATH=/weka/dfive-default/yawenz/datasets/scratch_ft_v0/dataset
export NUM_WORKERS=32
export TRAINER_DATA_PATH=/weka/dfive-default/yawenz/test/nandi
export WANDB_PROJECT=2025_10_08_nandi_crop_type
export WANDB_NAME=nandi_crop_type_segment_helios_base_S2_ts_ws1_ps1
export WANDB_ENTITY=eai-ai2
python -m rslp.main olmoearth_run olmoearth_run --config_path olmoearth_run_data/nandi/ --scratch_path /weka/dfive-default/yawenz/datasets/scratch_v5/ --checkpoint_path /weka/dfive-default/yawenz/test/checkpoints/last_rewritten.ckpt
```
Note that the original `task_name` has been converted from `crop_type_classification` to `class` in the checkpoint, and this checkpoint is also available at `gs://earth-system-run-dev/models/4edd1efb-b645-44c3-8d7a-5cc2abbbcc46/stage_0/checkpoint.ckpt`.
