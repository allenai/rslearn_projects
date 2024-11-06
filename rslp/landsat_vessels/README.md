Landsat Vessel Detection
------------------------

This detects vessels in Landsat imagery using two models:

1.  An object detector that detects fixed-size bounding boxes corresponding to vessels.
2.  A classifier that inputs small crops centered at detected vessels, and predicts
 whether the vessel is correct or not.


Training
--------

The object detector can be trained like this:

    python -m rslp.rslearn_main model fit --config data/landsat_vessels/config.yaml

The dataset was originally labeled in siv and has been converted to rslearn dataset
using the code in `landsat/existing_dataset_to_utm/`.

The classifier can be trained like this:

    python -m rslp.rslearn_main model fit --config landsat/recheck_landsat_labels/phase123_config.yaml

The data collection process for the classifier is described in
`landsat/recheck_landsat_labels/README.md`.


Prediction Pipeline
-------------------

First download the Landsat scene files, e.g. from USGS EarthExplorer or AWS.

Then create a configuration file for the prediction pipeline, here is an example:

```json
{
    "image_files": {
    "B2": "/home/favyenb/landsat_vessels_test_data/LC08_L1TP_125059_20240727_20240801_02_T1_B2.TIF",
    "B3": "/home/favyenb/landsat_vessels_test_data/LC08_L1TP_125059_20240727_20240801_02_T1_B3.TIF",
    "B4": "/home/favyenb/landsat_vessels_test_data/LC08_L1TP_125059_20240727_20240801_02_T1_B4.TIF",
    "B5": "/home/favyenb/landsat_vessels_test_data/LC08_L1TP_125059_20240727_20240801_02_T1_B5.TIF",
    "B6": "/home/favyenb/landsat_vessels_test_data/LC08_L1TP_125059_20240727_20240801_02_T1_B6.TIF",
    "B7": "/home/favyenb/landsat_vessels_test_data/LC08_L1TP_125059_20240727_20240801_02_T1_B7.TIF",
    "B8": "/home/favyenb/landsat_vessels_test_data/LC08_L1TP_125059_20240727_20240801_02_T1_B8.TIF",
    },
    "scratch_path": "/home/favyenb/landsat_vessels_test_data/scratch/",
    "json_path": "/home/favyenb/landsat_vessels_test_data/out/vessels.json",
    "crop_path": "/home/favyenb/landsat_vessels_test_data/out/crops/"
}
```

This specifies the arguments to
`rslp.landsat_vessels.predict_pipeline.predict_pipeline` via `jsonargparse`.

Now we can run the pipeline:

    python -m rslp.main landsat_vessels predict --config /path/to/config.json

Other alternatives:

- Run it with a path to the zipped Landsat scene files (downloaded locally or on GCS):

    `python -m rslp.main landsat_vessels predict --scene_zip_path /path/to/scene.zip --scratch_path /path/to/scratch/ --json_path /path/to/vessels.json --crop_path /path/to/crops/`

- Run it with a Landsat scene ID (to be fetched from AWS):

    `python -m rslp.main landsat_vessels predict --scene_id LC09_L1GT_106084_20241002_20241002_02_T2 --scratch_path /path/to/scratch/ --json_path /path/to/vessels.json --crop_path /path/to/crops/`


Evaluation
----------

To evaluate the model before integration, we prepared multiple scenes that cover different regions, failure modes (whitcaps, clouds, ice, islands, etc.), and true positives to check if the model is working correctlly. We can keep adding more evaluation scenes for a better coverage of different failure modes.

The evaluation process is as follows:

1. Launch the prediction jobs for the evaluation scenes:

`python rslp/landsat_vessels/job_launcher.py --zip_dir gs://rslearn-eai/projects/2024_10_check_landsat/evaluation/downloads/ --json_dir gs://rslearn-eai/projects/2024_10_check_landsat/evaluation/jsons/`

This will launch multiple beaker jobs. Each job will evaluate the model on one scene and save the results in the `jsons` directory.

2. Check the results against the targets (expected results) at scene level.

`python rslp/landsat_vessels/scripts/evaluate_predictions.py`

This will output the details of each scene (e.g. scene id, description, location, expected number of detections, actual number of detections), as well as the total number of passes and fails, and the success rate.
