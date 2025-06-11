This folder contains scripts to convert the Sentinel-1 vessel detection dataset from
sentinel-vessel-detection (siv format and WebMercator projection) to rslearn format
(and in UTM/UPS projections).

First, run the conversion script:

    python one_off_projects/convert_satlas_webmercator_to_rslearn/sentinel1_vessel/convert_siv_labels.py

This will convert the data, assuming dfive-default WEKA bucket is mounted at
`/weka/dfive-default`. It will use the metadata.sqlite3 (which has been copied to the
WEKA bucket) from sentinel-vessel-detection (`data/metadata.sqlite3`), this database
contains the vessel labels.

The conversion script creates windows with time range a few minutes around the time of
the Sentinel-1 image. But it may match multiple images, so then use
`delete_bad_images.py` to ensure that only the correct image is selected (i.e., the
image that matches the image ID from the original label in metadata.sqlite3).

Next, split the dataset into groups of ascending and descending windows:

    python one_off_projects/convert_satlas_webmercator_to_rslearn/split_ascending_descending.py --ds_path ...

Use `data/sentinel1_vessels/config.json` to prepare the ascending groups:

    cp data/sentinel1_vessels/config.json [dataset path]/config.json
    python -m rslp.rslearn_main dataset prepare --root [dataset_path] --workers 32 --groups X_ascending Y_ascending ...

Then update the `config.json` setting orbit_direction to DESCENDING and prepare the descending groups:

    python -m rslp.rslearn_main dataset prepare --root [dataset_path] --workers 32 --groups X_descending Y_descending ...

Then materialize:

    python -m rslp.rslearn_main dataset materialize --root [dataset_path] --workers 64
