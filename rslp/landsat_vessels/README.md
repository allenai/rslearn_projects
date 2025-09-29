# Finetune Helios for Landsat vessel detection

- Detector: 161747 train, 23824 val

Helios
```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_landsat_vessels/finetune_detector.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_landsat_vessel_detection_helios_base_ps4_freeze_unfreeze --gpus 1
```

SwinB pretrained on ImageNet
```
python -m rslp.main common beaker_train --image_name favyen/rslphelios2 --config_paths+=data/helios/v2_landsat_vessels/finetune_detector_swinb.yaml --cluster+=ai2/saturn-cirrascale --project_id 2025_06_26_helios_finetuning --experiment_id v2_landsat_vessel_detection_swin_imagenet_ps4 '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}' --gpus 1
```

- Classifier: 1783 train, 535 val

```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_landsat_vessels/finetune_classifier.yaml --cluster+=ai2/ceres-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_landsat_vessel_classification_helios_base_ps4_add_prob_threshold
```
