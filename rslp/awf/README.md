# African Wildlife Foundation LULC

There're in total 1469 points of 9 LULC classes in 2023.

| Class                          | Count |
|--------------------------------|-------|
| Agriculture / Settlement       | 288   |
| Grassland / Barren             | 320   |
| Herbaceous Wetland             | 49    |
| Lava Forest                    | 18    |
| Montane Forest                 | 59    |
| Open Water                     | 55    |
| Shrubland / Savanna            | 412   |
| Urban / Dense Development      | 90    |
| Woodland Forest (>40% canopy)  | 168   |

For initial experiment, we will use all the samples to finetune OlmoEarth for LULC classification.

Create datasets:
```
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/awf_2023
export DATASET_GROUP=20250822
python rslp/crop/awf/create_windows.py --csv_path=/weka/dfive-default/rslearn-eai/artifacts/AWF/Amboseli2023_lulc_training_v3.csv --ds_path=$DATASET_PATH
rslearn dataset prepare --root $DATASET_PATH --group $DATASET_GROUP --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
rslearn dataset materialize --root $DATASET_PATH --group $DATASET_GROUP --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
```

Launch finetune:
```
python -m rslp.main helios launch_finetune --image_name favyen/rslphelios10 --config_paths+=data/helios/v2_awf_lulc/finetune_s2_20250822.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_08_22_awf_lulc_classification --experiment_id awf_lulc_classification_helios_base_S2_ts_ws4_ps2_bs8
```

Prediction:
```
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/awf_2023
rslearn dataset add_windows --root $DATASET_PATH --group amboseli --utm --resolution 10 --grid_size 128 --src_crs EPSG:4326 --box=36.897854805569345,-3.1877100113094623,37.93955567770978,-2.2610410490685697 --start 2023-06-15T00:00:00+00:00 --end 2023-07-15T00:00:00+00:00 --name amboseli
```

7452 windows in total for inference

<!-- Original CRS: EPSG:4210
Min Lon: 36.897854805569345
Min Lat: -3.1877100113094623
Max Lon: 37.93955567770978
Max Lat: -2.2610410490685697 -->

```
python -m rslp.main helios launch_finetune --image_name favyen/rslphelios10 --config_paths+=data/helios/v2_awf_lulc/finetune_s2_20250822.yaml --cluster+=ai2/ceres-cirrascale --rslp_project 2025_08_22_awf_lulc_classification --experiment_id awf_lulc_classification_helios_base_S2_ts_ws4_ps2_bs8 --mode predict --gpus 4
```
