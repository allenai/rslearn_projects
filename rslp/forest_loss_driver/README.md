# Forest Loss Driver

## Overview
The Forest Loss Driver project aims to develop an automated global deforestation detection system to combat illegal deforestation activities. By leveraging satellite imagery and machine learning, this system:

- Detects and monitors forest loss events in near real-time
- Identifies the drivers/causes of deforestation (e.g. agriculture, mining, logging)
- Provides evidence through before/after satellite imagery
- Enables rapid response to illegal activities
- Supports conservation area managers in resource allocation
- Improves accountability and enforcement of forest protection laws

This technology is critical for preserving forests worldwide, protecting biodiversity, and mitigating climate change impacts. The system focuses particularly on monitoring protected areas and Indigenous territories where illegal deforestation remains a significant threat despite legal protections.

## Core Functionality

The system consists of two main pipeline components:

1. **Dataset Extraction Pipeline**
   - Processes forest loss alert GeoTIFFs to identify recent deforestation events
   - Collects satellite imagery (Sentinel-2, Planet) before and after each event
   - Filters for cloud-free images to ensure high quality data
   - Materializes an rslearn dataset with standardized windows around each event

2. **Model Prediction Pipeline**
   - Takes the prepared dataset and runs inference using trained models
   - Classifies the driver/cause of each deforestation event
   - Outputs predictions in GeoJSON format with confidence scores
   - Supports batch processing for large-scale inference

## Usage

### Environment Setup
Required environment variables:
- `RSLP_PREFIX`: GCS bucket prefix for model checkpoints \

Optional environment variables:
- `INDEX_CACHE_DIR`: Directory for caching image indices MUST SPECIFY FILE SYSTEM OR IT WILL BE TREATED ad relative path
- `TILE_STORE_ROOT_DIR`: Directory for tile storage cache
- `PL_API_KEY`: Planet API key (if using Planet imagery)

Otherwise, follow set up in [main readme](../../README.md)

### Pipeline Configuration

The current inference data configuration is stored in [data/forest_loss_driver/config.json](../../data/forest_loss_driver/config.json). This contains the bands and data sources the model needs to perform inference. It is essential this dataset configuration matches the configuration used to train the model.

The current pipeline configuration is stored in [forest_loss_driver_predict_pipeline_config.yaml](inference/config/forest_loss_driver_predict_pipeline_config.yaml) the default values can be found in this [config class](inference/config.py). This configuration points to the model configuration currently in use by the pipeline.
### Running the Pipeline

1. Extract dataset

2. Predict Forest Loss Driver Events


## Links to other specific kinds of docs and functionality

training doc \
deployment doc \

## Adding Examples to ES Studio

Here are the steps for adding forest loss driver classification tasks in Brazil and
Colombia to ES Studio.

First, run the alert extraction pipeline:

   python -m rslp.main forest_loss_driver extract_alerts --ds_path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --extract_alerts_args.gcs_tiff_filenames '["080W_00N_070W_10N.tif", "080W_10S_070W_00N.tif", "070W_10S_060W_00N.tif", "070W_00N_060W_10N.tif"]' --extract_alerts_args.countries '["CO"]' --extract_alerts_args.tile_store_dir "file:///weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/tile_store_root_dir/" --extract_alerts_args.index_cache_dir "file:///tmp/index_cache_dir/" --extract_alerts_args.workers 128 --extract_alerts_args.max_number_of_events 5000 --extract_alerts_args.group 20250428_colombia --extract_alerts_args.days 1095 --extract_alerts_args.prediction_utc_time "2025-03-01 00:00:00+00:00"
   python -m rslp.main forest_loss_driver extract_alerts --ds_path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --extract_alerts_args.gcs_tiff_filenames '["050W_20S_040W_10S.tif", "060W_20S_050W_10S.tif", "070W_20S_060W_10S.tif", "040W_10S_030W_00N.tif", "050W_10S_040W_00N.tif", "060W_10S_050W_00N.tif", "070W_10S_060W_00N.tif", "080W_10S_070W_00N.tif", "050W_00N_040W_10N.tif", "060W_00N_050W_10N.tif", "070W_00N_060W_10N.tif", "080W_00N_070W_10N.tif"]' --extract_alerts_args.countries '["BR"]' --extract_alerts_args.tile_store_dir "file:///weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/tile_store_root_dir/" --extract_alerts_args.index_cache_dir "file:///tmp/index_cache_dir/" --extract_alerts_args.workers 128 --extract_alerts_args.max_number_of_events 5000 --extract_alerts_args.group 20250428_brazil --extract_alerts_args.days 1095 --extract_alerts_args.prediction_utc_time "2025-03-01 00:00:00+00:00"

Switch the rslearn dataset configuration file with the one in
`data/forest_loss_driver/config_rgb_geotiff.json`. This obtains an 8-bit RGB GeoTIFF
for the Sentinel-2 data, along with Planet Labs imagery. Then run the standard prepare,
ingest, and materialize steps.

Use this script to populate a label layer:

```python
import json
import multiprocessing

import tqdm
from upath import UPath

def process_window(window_dir):
    with (window_dir / "layers" / "mask_vector" / "data.geojson").open() as f:
        fc = json.load(f)
        assert len(fc["features"]) == 1
    fc["features"][0]["properties"]["old_label"] = "unknown"
    fc["features"][0]["properties"]["new_label"] = "unknown"
    dst_fname = window_dir / "layers" / "label" / "data.geojson"
    dst_fname.parent.mkdir(parents=True, exist_ok=True)
    with dst_fname.open("w") as f:
        json.dump(fc, f)
    (dst_fname.parent / "completed").touch()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    ds_path = UPath("/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/")
    window_dirs = list(ds_path.glob("windows/*/*"))
    p = multiprocessing.Pool(64)
    outputs = p.imap_unordered(process_window, window_dirs)
    for _ in tqdm.tqdm(outputs, total=len(window_dirs)):
        pass
    p.close()
```

Now import into ES Studio:

   python tools/rslearn_import.py --dataset-path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --layers best_pre_0 best_pre_1 best_pre_2 best_post_0 best_post_1 best_post_2 label planet_monthly --api-url https://earth-system-studio.allen.ai --project-name 'Forest Loss Driver Brazil 7' --always-upload-rasters --workers 64 --groups 20250428_brazil_phase1
   python tools/rslearn_import.py --dataset-path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --layers best_pre_0 best_pre_1 best_pre_2 best_post_0 best_post_1 best_post_2 label planet_monthly --api-url https://earth-system-studio.allen.ai --project-name 'Forest Loss Driver Colombia 7' --always-upload-rasters --workers 64 --groups 20250428_colombia_phase1
