```shell
export PROJECT_PATH=/weka/dfive-default/hadriens/olmoearth_projects/docs/tutorials/FinetuneOlmoEarthSegmentation/config
export OER_DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/burn-scar/oerun_raster
export WANDB_PROJECT="oe_burn-scar-finetuning"
export WANDB_NAME="burn-scar_seg_s2_p4_c128_unet_lr1e4"
export WANDB_ENTITY="eai-ai2"
export NUM_WORKERS=30
#export TRAINER_DATA_PATH="/weka/dfive-default/rslearn-eai/datasets/burn-scar/oerun_raster/dataset/trainer_checkpoints"

python -m olmoearth_projects.main olmoearth_run finetune \
  --project_path $PROJECT_PATH \
  --scratch_path $OER_DATASET_PATH
```


```shell
unset TRAINER_DATA_PATH
unset DATASET_PATH
unset EXTRA_FILES_PATH
export CHECKPOINT_PATH=/weka/dfive-default/rslearn-eai/datasets/burn-scar/oerun_raster/trainer_checkpoints/last.ckpt
#export EXTRA_FILES_PATH=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200
python -m olmoearth_projects.main olmoearth_run olmoearth_run \
  --config_path $PROJECT_PATH \
  --scratch_path $OER_DATASET_PATH \
  --checkpoint_path $CHECKPOINT_PATH
```
