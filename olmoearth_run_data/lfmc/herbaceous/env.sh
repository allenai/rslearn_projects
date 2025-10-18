# Run source env.sh to set the environment variables
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/lfmc/20251015-herbaceous-scratch/dataset
export EXTRA_FILES_PATH=/weka/dfive-default/olmoearth_pretrain/checkpoints
export NUM_WORKERS=32
export WANDB_ENTITY=eai-ai2
export WANDB_NAME=lfmc_olmoearth_herbaceous_$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 4)
export WANDB_PROJECT=$(date +%Y_%m_%d)_lfmc
