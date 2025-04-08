Dataset
-------

The dataset is created from CSV files of AIS-correlated vessel detections provided by
Skylight.

To populate the dataset:

```
mkdir /path/to/dataset/
cp data/sentinel2_vessel_attribute/config.json /path/to/dataset/config.json
python -m rslp.main sentinel2_vessel_attribute create_windows --group detections_bigtable --csv_dir gs://rslearn-eai/datasets/sentinel2_vessel_attribute/artifacts/sentinel2_correlated_detections_bigtable/ --ds_path gs://rslearn-eai/datasets/sentinel2_vessel_attribute/dataset_v1/20250205/ --workers 64
python -m rslp.main sentinel2_vessel_attribute create_windows --group detections_jan_470k --csv_dir gs://rslearn-eai/datasets/sentinel2_vessel_attribute/artifacts/sentinel2_correlated_detections_jan_470k/ --ds_path gs://rslearn-eai/datasets/sentinel2_vessel_attribute/dataset_v1/20250205/ --workers 64
```

This puts the first batch of CSVs in one group ("detections_bigtable") and the second
batch in another group.
