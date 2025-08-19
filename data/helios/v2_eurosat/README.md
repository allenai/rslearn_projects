This compares Helios model to Google Satellite Embeddings on EuroSat.

Note that some EuroSat examples are skipped because we weren't able to get GSE
embeddings for them. It should be consistent across `helios.yaml` and `gse.yaml` since
we mark the `gse` layer required for both of them.

```
# Google Satellite Embeddings
python -m rslp.main common beaker_train --config_paths+=data/helios/v2_eurosat/gse.yaml --project_id 2025_07_30_eurosat_comparison --experiment_id gse_frozen --cluster+=ai2/ceres-cirrascale --cluster+=ai2/jupiter-cirrascale-2 --cluster+=ai2/titan-cirrascale '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}' --image_name favyen/rslphelios10


# Helios
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_naip_moredata_random_fixed_modality_0.5/step320000 --patch_size 8 --encoder_embedding_size 768 --image_name favyen/rslphelios10 --config_paths+=data/helios/v2_eurosat/helios.yaml --cluster+=ai2/ceres-cirrascale --cluster+=ai2/jupiter-cirrascale-2 --cluster+=ai2/titan-cirrascale --rslp_project 2025_07_30_eurosat_comparison --experiment_id helios_alldata_320k_ps8_frozen
```
