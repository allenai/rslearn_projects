# Kenya/Nandi County Crop Type Classification

The original dataset contains 819 polygons, which has been converted into about 19K points (at 10m resolution). The converted points include point_id, polygon_id, latitude (y), longitude (x), category, and other metadata (like planting date, harvest date).

- Polygons: `gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/NandiGroundTruth`
- Points: `gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/NandiGroundTruthPoints.csv`

The original categories didn’t include water or built-up areas. To support Land Use and Land Cover (LULC) mapping, we added randomly sampled points from [WorldCover](https://viewer.esa-worldcover.org/worldcover/) for "Water" (Value: 50 in WorldCover) and "Built-up" (Value: 80 in WorldCover) classes.

- Kenya/Nandi county shapefile: `gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/Nandi_County`
- ESA WorldCover images for Nandi county: `gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/WorldCover/NandiCounty_worldcover.tif`

### Step 1. Create Windows

Run the command to create windows for the groundtruth points:
```
python rslp/crop/kenya_nandi/create_windows_for_groundtruth.py --csv_path=gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/NandiGroundTruthPoints.csv --ds_path=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625 --window_size=32
```

By default, we sample at most 10 pixels per polygon, to avoid the case where one polygon creates many homogeneous points with the same category. Also, following the CGIAR/IFPRI workflow (more details can be found [here](https://www.ifpri.org/blog/from-space-to-soil-advancing-crop-mapping-and-ecosystem-insights-for-smallholder-agriculture-in-kenya/)), we can optionally apply postprocessing on the original categories, by merging the "Exoticetrees/forests" and "Nativetrees/forest" into "Trees", and dropping the categories with less labels, mainly "Legumes" and "Vegatables". By default, we keep the original categories (9 classes in total).

<img src="figures/categories.png" alt="Categories" width="40%">


Run the command to create windows for the worldcover points (we sampled 1K points for Water and Built-up separately):
```
python rslp/crop/kenya_nandi/create_windows_for_worldcover.py --csv_path=gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/NandiWorldCoverPoints_sampled.csv --ds_path=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625 --window_size=32
```

- rslearn dataset: `/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250616`
- GroundTruth group: `groundtruth_random_split_window_32`
- WorldCover group: `worldcover_window_32`

TODO(yawenz): add grid split, ensure that nearby polygons are not in the same split.

### Step 2. Prepare/Materialize Windows

- Data Configuration: `/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250616/config.json`

For this task, we primarily use Sentinel-1 and Sentinel-2 L2A data, selecting the most recent 6 months (as defined by `max_matches` in the data configuration) from the 1-year data.

Run the command to prepare and materialize groundtruth windows:
```
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625 --group groundtruth_polygon_split_window_32 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

rslearn dataset materialize --root /weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625 --group groundtruth_polygon_split_window_32 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
```

Run the command to prepare and materialize worldcover windows:
```
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625 --group worldcover_window_32 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

rslearn dataset materialize --root /weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625 --group worldcover_window_32 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
```

### Step 3. Finetune Helios

- Helios Checkpoint: `/weka/dfive-default/helios/checkpoints/joer/v0.1_base_latent_mim_space_time/step165000`
- Model Configuration: `data/helios/v2_nandi_crop_type/finetune_s1_s2.yaml`

Run the command to start finetuning Helios for crop type classification:

```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 1 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_nandi_crop_type/finetune_s1_s2.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_crop_type_classification_helios_base_S1_S2_ts_ws1_ps1
```

```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 1 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_nandi_crop_type/finetune_s2.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_crop_type_classification_helios_base_S2_ts_ws2_ps1
```

Experiments:

- Use polygon split, since we need to generate maps for the whole Nandi county
- Ready for inference: window_size = 1, patch_size = 1 (run this one first, then prediction pipeline)
- Comparison (window_size, patch_size): (32, 8), (8, 2), (4, 1), (1, 1)


### Step 4. Make Maps!

```
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250616
rslearn dataset add_windows --root $DATASET_PATH --group nandi_county --utm --resolution 10 --grid_size 128 --src_crs EPSG:4326 --box=34.6999,-0.114,35.4549,0.5672 --start 2023-03-01T00:00:00+00:00 --end 2023-03-31T00:00:00+00:00 --name nandi
```

Split by grids, right now, we're seeing better performance with the window size 32, but it could just be some data leakage, as some polygons may be closer to each other, especially the one with the same category.

Prediction
```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 2 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_nandi_crop_type/finetune_s2.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_crop_type_classification_helios_base_S2_ts_ws2_ps2 --mode predict
```
