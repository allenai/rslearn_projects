Local esrun for Nandi
```bash
export EXTRA_FILES_PATH=/weka/dfive-default/helios/checkpoints
export DATASET_PATH=/weka/dfive-default/yawenz/datasets/scratch_v5/dataset_0
export NUM_WORKERS=32
export TRAINER_DATA_PATH=/weka/dfive-default/yawenz/test/nandi
export WANDB_PROJECT=2025_10_03_nandi_crop_type
export WANDB_NAME=nandi_crop_type_segment_helios_base_S2_S1_ts_ws4_ps1_bs8_add_annotations_2
export WANDB_ENTITY=eai-ai2
python -m rslp.main esrun esrun --config_path esrun_data/nandi/ --scratch_path /weka/dfive-default/yawenz/datasets/scratch_v5/ --checkpoint_path /weka/dfive-default/yawenz/test/checkpoints/last_rewritten.ckpt
```
