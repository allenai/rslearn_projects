# Environment Setup

```bash
export EXTRA_FILES_PATH=/weka/dfive-default/helios/checkpoints
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/lfmc/20251007-scratch/dataset
export NUM_WORKERS=32
export TRAINER_DATA_PATH=/weka/dfive-default/patrickj/test/lfmc
export WANDB_PROJECT=2025_10_08_lfmc
export WANDB_NAME=lfmc_helbios_base_s2_ts_ws1_ps1
export WANDB_ENTITY=eai-ai2
```

# Fine-tune
```bash
python -m rslp.main esrun finetune --project_path /weka/dfive-default/patrickj/rslearn_projects/esrun_data/lfmc --scratch_path /weka/dfive-default/rslearn-eai/datasets/lfmc/20251007-scratch
```
