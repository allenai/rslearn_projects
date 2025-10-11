# Run source env.sh to set the environment variables

export EXTRA_FILES_PATH=/weka/dfive-default/helios/checkpoints

export WANDB_PROJECT=$(date +%Y_%m_%d)_lfmc
export WANDB_NAME=lfmc_olmoearth_base_s1_s2_$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 4)
export WANDB_ENTITY=eai-ai2
