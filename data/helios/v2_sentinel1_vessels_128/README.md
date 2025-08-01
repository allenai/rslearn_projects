The configs are meant to be used like this:

- basecfg.yaml + basecfg_swinb.yaml (random)
- basecfg.yaml + basecfg_swinb.yaml + imagenet_swinb.yaml (ImageNet)
- basecfg.yaml + basecfg_helios.yaml + helios_freeze_then_lowlr.yaml (Helios)

```
# swinb_random
python -m rslp.main common beaker_train --config_paths+=data/helios/v2_sentinel1_vessels_128/basecfg.yaml --config_paths+=data/helios/v2_sentinel1_vessels_128/basecfg_swinb.yaml --project_id 2025_06_06_helios_finetuning --experiment_id v2_sentinel1_vessels_128_swinb_random --cluster+=ai2/ceres-cirrascale --cluster+=ai2/saturn-cirrascale '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}' --image_name favyen/rslphelios2

# swinb_imagenet
python -m rslp.main common beaker_train --config_paths+=data/helios/v2_sentinel1_vessels_128/basecfg.yaml --config_paths+=data/helios/v2_sentinel1_vessels_128/basecfg_swinb.yaml --config_paths+=data/helios/v2_sentinel1_vessels_128/imagenet_swinb.yaml --project_id 2025_06_06_helios_finetuning --experiment_id v2_sentinel1_vessels_128_swinb_imagenet --cluster+=ai2/ceres-cirrascale --cluster+=ai2/saturn-cirrascale '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}' --image_name favyen/rslphelios2

# Helios
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/joer/v0.1_base_latent_mim_space_time/step165000 --patch_size 8 --encoder_embedding_size 768 --image_name favyen/rslphelios2 --config_paths+=data/helios/v2_sentinel1_vessels_128/basecfg.yaml --config_paths+=data/helios/v2_sentinel1_vessels_128/basecfg_helios.yaml --config_paths+=data/helios/v2_shared/helios_freeze_then_lowlr.yaml --cluster+=ai2/ceres-cirrascale --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_06_helios_finetuning --experiment_id v2_sentinel1_vessels_128_helios_latent_mim_space_time
```
