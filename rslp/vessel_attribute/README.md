This module contains shared code for training vessel attribute prediction models for
Sentinel-2, Landsat, and Sentinel-1.


Creating Windows
----------------

The dataset is created from CSV files of AIS-correlated vessel detections provided by
Skylight.

To populate the dataset for Sentinel-2:

```
mkdir /path/to/dataset/
cp data/sentinel2_vessel_attribute/config.json /path/to/dataset/config.json
python -m rslp.main vessel_attribute create_windows --group detections_bigtable --csv_dir gs://rslearn-eai/datasets/sentinel2_vessel_attribute/artifacts/sentinel2_correlated_detections_bigtable/ --ds_path gs://rslearn-eai/datasets/sentinel2_vessel_attribute/dataset_v1/20250205/ --workers 64
python -m rslp.main vessel_attribute create_windows --group detections_jan_470k --csv_dir gs://rslearn-eai/datasets/sentinel2_vessel_attribute/artifacts/sentinel2_correlated_detections_jan_470k/ --ds_path gs://rslearn-eai/datasets/sentinel2_vessel_attribute/dataset_v1/20250205/ --workers 64
```

This puts the first batch of CSVs in one group ("detections_bigtable") and the second
batch in another group.

For Sentinel-1 and Landsat, see the source CSVs here:
- /weka/dfive-default/rslearn-eai/datasets/landsat_vessel_attribute/dataset_v1/20260330/source_csvs/landsat_correlated_detetions_20260329.csv
- /weka/dfive-default/rslearn-eai/datasets/sentinel1_vessel_attribute/dataset_v1/20260330/source_csvs/sentinel1_correlated_detections_20260329.csv


Training
--------

`train.py` contains training code to handle things like predicting the heading (and
handling heading during flip augmentation), creating consolidated visualizations for
all the attribute prediction tasks when using `rslearn model test`, etc.

The dataset and model configs are in per-modality folders:
- data/sentinel2_vessel_attribute/
- data/landsat_vessel_attribute/
- data/sentinel1_vessel_attribute/


Evaluation
----------

There are a few scripts in `scripts/` that rely on a prediction CSV. First output the
predictions like this:

```python
rslearn model predict --config data/landsat_vessel_attribute/20260422/config_new.yaml --data.init_args.predict_config.groups='["default", "20260421"]' --data.init_args.predict_config.tags='{"split": "val"}'
```

Then create the prediction CSV:

```python
python -m rslp.vessel_attribute.scripts.predictions_to_csv --ds_path /weka/dfive-default/rslearn-eai/datasets/landsat_vessel_attribute/dataset_v1/20260330/ --output /weka/dfive-default/rslearn-eai/datasets/landsat_vessel_attribute/dataset_v1/20260330/output.csv
```

The CSV can then be used for analytics, e.g. computing confusion matrix per length bucket:

```python
python -m rslp.vessel_attribute.scripts.confusion_by_length --csv /weka/dfive-default/rslearn-eai/datasets/landsat_vessel_attribute/dataset_v1/20260330/output.csv
```
