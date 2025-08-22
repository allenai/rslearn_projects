Experiment 1: Single modality (Sentinel2) + time-series (12 months), window_size = 32, patch_size = 8
```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 8 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_lfmc/finetune_s2.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_lfmc_helios_base_S2_ts_ws32_ps8
```

Experiment 2: Multimodal (Sentinel2 + Sentinel1 + SRTM) + time-series (12 months), window_size = 32, patch_size = 8
```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 8 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_lfmc/finetune_s1_s2_srtm.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_lfmc_helios_base_S1_S2_SRTM_ts_ws32_ps8
```

Experiment 3: Random initialized Helios (same setup as Experiment 2)
```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 8 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_lfmc/random_s1_s2_srtm.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_lfmc_helios_base_random_initialized_ws32_ps8
```
