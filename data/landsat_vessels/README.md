Dataset Configurations
----------------------

The same dataset configuration file is used for the detector and the classifier.

- predict_dataset_config.json: for the prediction pipeline, when using local GeoTIFFs.
- predict_dataset_config_aws.json: for the prediction pipeline, when getting Landsat
  images from AWS.
- train_dataset_config.json: for training. The difference with
  predict_dataset_config_aws.json is just that this one uses an external tile store
  directory.

For `train_dataset_config.json`, the tile store is a local directory. It is intended to
run ingestion and materialization on AWS so the full scenes never need to be uploaded
off of AWS, only the crops pertaining to the dataset windows.

For the prediction datasets, the dataset configs include a "landsat_allbands" layer
that is used for the vessel attribute predictor (the detector only uses a subset of
bands, and expects them to have remapped values, while the attribute predictor expects
all bands with their raw values).


Dataset Versions
----------------

- 20250206: same as classifier dataset 20240905 and detector dataset 20240924, but we
  add resampling_method=nearest to the dataset configuration file so the dataset needed
  to be materialized from scratch.
