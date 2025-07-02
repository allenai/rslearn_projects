# Live Fuel Moisture Content (LFMC) Estimation

The original LFMC work is based on Galileo model (more details can be found in Patrick J's [paper](https://www.overleaf.com/project/6819d1e76eb0ba2dde451e04), TODO: update link once it's opne-sourced) and the data preparation part is relying on Google Earth Engine. Here, we switch to rslearn for data preparation (right now, only the following modalities are supported: Sentinel2, Sentinel1, SRTM, location, we will need to add ERA5 and mayb TerraClimate in the future). The alation study of the paper (Table 9) indicates that adding ERA5 and TerraClimate help reduce the error in LFMC estimation.

- LFMC ground-truth data: `/weka/dfive-default/yawenz/datasets/LFMC/lfmc-labels.csv`

This file is the processed LFMC ground-truth data, which includes metadata like `latitude`, `longitude`, `sampling_date`, `site_name`, `lfmc_value`, `state_region`, `country`, `landcover` and `elevation`. We further cut off the LFMC value by 302 which is the 99.9% value.

### 1. Create Windows

Run the following commands to create windows:
```
python rslp/lfmc/create_windows.py --csv_path /weka/dfive-default/yawenz/datasets/LFMC/lfmc-labels.csv --ds_path /weka/dfive-default/rslearn-eai/datasets/lfmc/20250626 --window_size 32
```

### 2. Prepare/Ingest/Materialize Windows

Run the command to prepare/ingest/materialize groundtruth windows (the ingestion is mainly for SRTM):
```
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/lfmc/20250626
export DATASET_GROUP=global_lfmc
rslearn dataset prepare --root DATASET_PATH --group DATASET_GROUP --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
rslearn dataset ingest --root DATASET_PATH --group DATASET_GROUP --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
python rslp/scripts/beaker_launcher.py --project lfmc --ds_path DATASET_PATH --group DATASET_GROUP --image_name favyen/rslp --clusters ai2/saturn-cirrascale --num_jobs 10
```

Note that to run these commands, we will need to set the environmental variables of `NASA_EARTHDATA_USERNAME` and `NASA_EARTHDATA_PASSWORD` ([Link](https://urs.earthdata.nasa.gov/)) for accessing SRTM data.

### 3. Finetune Helios

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
