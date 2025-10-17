Local olmoearth_run finetune for Nandi:
```bash
export EXTRA_FILES_PATH=/weka/dfive-default/helios/checkpoints
export DATASET_PATH=/weka/dfive-default/yawenz/datasets/scratch_ft_v2/dataset
export NUM_WORKERS=32
export TRAINER_DATA_PATH=/weka/dfive-default/yawenz/test/nandi
export WANDB_PROJECT=2025_10_16_nandi_crop_type
export WANDB_NAME=nandi_crop_type_segment_helios_base_S2_ts_ws32_ps4
export WANDB_ENTITY=eai-ai2
python -m rslp.main olmoearth_run prepare_labeled_windows --project_path /weka/dfive-default/yawenz/rslearn_projects/olmoearth_run_data/nandi/finetune --scratch_path /weka/dfive-default/yawenz/datasets/scratch_ft_v2
python -m rslp.main olmoearth_run finetune --project_path /weka/dfive-default/yawenz/rslearn_projects/olmoearth_run_data/nandi/finetune --scratch_path /weka/dfive-default/yawenz/datasets/scratch_ft_v2
```
