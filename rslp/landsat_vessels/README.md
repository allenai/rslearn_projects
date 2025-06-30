```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_landsat_vessels/finetune_classifier.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_landsat_vessel_classification_helios_base_ps4
```

Classifier: 1783 train, 535 val

```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_landsat_vessels/finetune_detector.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_landsat_vessel_detection_helios_base_ps4 --gpus 4
```

Detector: 161747 train, 23824 val

- Experiments: patch_size (4, 8)
- Checkpoints: latentmin_space_time / latentmin_new_decode


/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000
/weka/dfive-default/helios/checkpoints/joer/v0.2_latent_mim_128_space_time_r2/step215000 (use rslphelios2)

Random initialized Helios:

```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_landsat_vessels/random_classifier.yaml --cluster+=ai2/ceres-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_landsat_vessel_classification_helios_base_random_initialized_ps4
```

```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_landsat_vessels/random_detector.yaml --cluster+=ai2/ceres-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_landsat_vessel_detection_helios_base_random_initialized_ps4_fix --gpus 4
```
