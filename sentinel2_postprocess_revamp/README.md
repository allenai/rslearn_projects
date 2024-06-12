Data
----

The input data for this project component are the CSV files that Hunter created
containing Sentinel-2 vessel position + AIS metadata.

There are currently two formats:

- First v1 CSV (`sentinel2_vessel_labels_with_metadata.csv`) with metadata for a subset of
  the vessel labels in the Sentinel-2 vessel detection dataset.

- Later v2 CSVs with metadata for recent AIS-correlated vessel predictions.

We get image crops centered at vessels using rslearn.
First create an empty dataset e.g. at `/data/favyenb/rslearn_sentinel2_vessel_postprocess`.

Copy `config.json` to the dataset directory.

Then create windows from the different CSVs:

    python create_windows_v1.py /home/favyenb/sentinel2_vessel_labels_with_metadata.csv /data/favyenb/rslearn_sentinel2_vessel_postprocess labels
    python create_windows_v2.py /home/favyenb/sentinel2_correlated_detections.csv /data/favyenb/rslearn_sentinel2_vessel_postprocess detections

Then use rslearn to prepare, ingest, and materialize the dataset:

    python -m rslearn.main dataset prepare --root /data/favyenb/rslearn_sentinel2_vessel_postprocess/ --workers 64 --group detections --batch-size 8
    python -m rslearn.main dataset ingest --root /data/favyenb/rslearn_sentinel2_vessel_postprocess/ --workers 64 --group detections --no-use-initial-job --jobs-per-process 1
    python -m rslearn.main dataset materialize --root /data/favyenb/rslearn_sentinel2_vessel_postprocess/ --workers 64 --group detections --no-use-initial-job --jobs-per-process 1

The materialized dataset as of 2024-06-12 (17K vessels) is available here:

    gs://satlas-explorer-data/rslearn_labels/rslearn_sentinel2_vessel_postprocess_2024-06-12.tar


Training
--------

`train_config.yaml` provides the configuration to train the model with rslearn.
The only custom component needed is a visualization function provided by `rslearn_entrypoint.py`.

Train the model:

    PYTHONPATH=/path/to/rslearn python rslearn_entrypoint.py model fit --config ../rslearn_projects/sentinel2_postprocess_revamp/train_config.yaml
    mkdir vis
    PYTHONPATH=/path/to/rslearn python rslearn_entrypoint.py model test --config ../rslearn_projects/sentinel2_postprocess_revamp/train_config.yaml --ckpt_path lightning_logs/version_0/checkpoints/... --model.visualize_dir vis/
