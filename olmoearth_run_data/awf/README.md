Local olmoearth_run finetune for AWF:
```bash
export EXTRA_FILES_PATH=/weka/dfive-default/helios/checkpoints
export DATASET_PATH=/weka/dfive-default/yawenz/datasets/awf_ft_v0/dataset
export NUM_WORKERS=32
export TRAINER_DATA_PATH=/weka/dfive-default/yawenz/test/awf
export WANDB_PROJECT=2025_10_25_awf_land_cover
export WANDB_NAME=awf_land_cover_segment_helios_base_S2_ts_ws16_ps1_bs4
export WANDB_ENTITY=eai-ai2
python -m rslp.main olmoearth_run prepare_labeled_windows --project_path /weka/dfive-default/yawenz/rslearn_projects/olmoearth_run_data/awf --scratch_path /weka/dfive-default/yawenz/datasets/awf_ft_v0
# Check train/val split
export GROUP_PATH=/weka/dfive-default/yawenz/datasets/scratch_ft_v3/dataset/windows/spatial_split
find $GROUP_PATH -maxdepth 2 -name "metadata.json" -exec cat {} \; | grep -oE "train|val|test" | sort | uniq -c | awk 'BEGIN{printf "{"} {printf "%s\"%s\": %d", (NR>1?", ":""), $2, $1} END{print "}"}'
python -m rslp.main olmoearth_run finetune --project_path /weka/dfive-default/yawenz/rslearn_projects/olmoearth_run_data/nandi/finetune --scratch_path /weka/dfive-default/yawenz/datasets/scratch_ft_v3
```
