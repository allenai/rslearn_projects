# Landsat Vessel Detection


## Training

This detects vessels in Landsat imagery using two models:

1.  An object detector that detects fixed-size bounding boxes corresponding to vessels.
2.  A classifier that inputs small crops centered at detected vessels, and predicts
 whether the vessel is correct or not.


The object detector can be trained like this:

    rslearn model fit --config data/landsat_vessels/config_detector.yaml

The dataset was originally labeled in siv and has been converted to rslearn dataset
using the code in `landsat/existing_dataset_to_utm/`.

The classifier can be trained like this:

    rslearn model fit --config data/landsat_vessels/config_classifier_20260908d.yaml

The current classifier is an OlmoEarth-base model (Run d) trained with round-1 annotations.
The round-1 annotation and data-collection process is described in
`rslp/landsat_vessels/annotations/README.md`, and the training experiments (runs a-g) are
summarized in `rslp/landsat_vessels/README.md`.

---

## Prediction Pipeline


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

Other options to run the pipeline:

Run it with a path to the zipped Landsat scene files (downloaded locally or on GCS):

    python -m rslp.main landsat_vessels predict --scene_zip_path /path/to/scene.zip --scratch_path /path/to/scratch/ --json_path /path/to/vessels.json --crop_path /path/to/crops/

Run it with a Landsat scene ID (to be fetched from AWS):

    python -m rslp.main landsat_vessels predict --scene_id scene_id --scratch_path /path/to/scratch/ --json_path /path/to/vessels.json --crop_path /path/to/crops/

Run it with a path to a window containing the metadata.json files:

    python -m rslp.main landsat_vessels predict --window_path /path/to/window/ --scratch_path /path/to/scratch/ --json_path /path/to/vessels.json --crop_path /path/to/crops/

---

## Evaluation

The whole pipeline is evaluated with two approaches.

- **Evaluation Metrics**: Evaluate the pipeline on the validation set of the detector (about 1K images), which outputs the recall, precision, and F1 score.
- **Scenario Checks**: Evaluate the pipeline on a set of selected scenes, which covers different regions, failure modes (whitcaps, clouds, ice, islands, etc.), and true positives, to validate if the pipeline is working properly.


### Evaluation Metrics

1. Launch the prediction jobs for the detector validation set:

    ```python
    python rslp/landsat_vessels/job_launcher.py --window_dir gs://rslearn-eai/datasets/landsat_vessel_detection/detector/dataset_20240924/windows/labels_utm/ --json_dir gs://rslearn-eai/projects/landsat_evaluation/pipeline_results/jsons/
    ```

This will launch multiple beaker jobs. Each job will evaluate the model on one window and save the results in the `jsons` directory.

2. Compute the metrics:

    ```python
    python rslp/landsat_vessels/evaluation/get_metrics.py --ground_truth_dir gs://rslearn-eai/datasets/landsat_vessel_detection/detector/dataset_20240924/windows/labels_utm --predictions_dir gs://rslearn-eai/projects/landsat_evaluation/pipeline_results/jsons/
    ```

This will output the evaluation metrics, including precision, recall, and F1 score.

### Scenario Checks (Smoke Test)

The scenario checks are run as a smoke test over a fixed set of scenes covering different
regions, failure modes (whitecaps, clouds, ice, islands), and true positives. The scenes
and their expected detection-count ranges are defined in
`rslp/landsat_vessels/evaluation/smoke_test.py`.

Run the pipeline on those scenes and check the counts against the expected ranges across a
detector x classifier threshold grid (prints a pass/fail matrix plus per-scene and total
detection-count matrices):

```python
python -m rslp.landsat_vessels.evaluation.smoke_test_sweep_2d --classify_config data/landsat_vessels/config_classifier_20260908d.yaml --out_dir /tmp/smoke_sweep_2d
```

To visualize the predictions (per-scene overview + pan-sharpened detection crops):

```python
python -m rslp.landsat_vessels.evaluation.visualize_smoke_predictions --classify_config data/landsat_vessels/config_classifier_20260908d.yaml --det_thr 0.7 --cls_thr 0.99 --out_dir /tmp/smoke_vis
```

See `rslp/landsat_vessels/README.md` for the latest smoke-test results and threshold sweep.
