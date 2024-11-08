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

Here, `scratch_path` is used to save the rslearn dataset, `crop_path` is used to save the cropped images, `json_path` is used to save the JSON output, all of which are optional, depending on whether the user wants to save the intermediate results or not.

Now we can run the pipeline:

    python -m rslp.main landsat_vessels predict --config /path/to/config.json

Alternatively, run it with a Landsat scene ID (to be fetched from AWS):

    python -m rslp.main landsat_vessels predict --scene_id LC09_L1GT_106084_20241002_20241002_02_T2 /path/to/scratch/ /path/to/vessels.json /path/to/crops/


API
---

First, we need to define the environment variables `LANDSAT_HOST` and `LANDSAT_PORT` to define the host and port of the Landsat service.

Then we can run the API server via `python rslp/landsat_vessels/api_main.py`. Sample request can be found in `rslp/landsat_vessels/scripts/sample_request.py`.
