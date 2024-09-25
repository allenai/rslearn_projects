Landsat Vessel Detection
------------------------

TODO



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
    "csv_path": "/home/favyenb/landsat_vessels_test_data/out/vessels.csv",
    "crop_path": "/home/favyenb/landsat_vessels_test_data/out/crops/"
}
```

This specifies the arguments to
`rslp.landsat_vessels.predict_pipeline.predict_pipeline` via `jsonargparse`.

Now we can run the pipeline:

    python -m rslp.main landsat_vessels predict_pipeline --config /path/to/config.json

Status:
* Currently only the detector is working. The classifier will fail.
* And really the detector isn't performing very well.
