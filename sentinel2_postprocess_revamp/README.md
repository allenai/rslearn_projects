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
