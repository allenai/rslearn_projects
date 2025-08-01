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

## Overview

This project consists of several components:

- Dataset extraction: the `rslp.forest_loss_driver.extract_dataset` module contains
  code for creating and materializing an rslearn dataset based on GLAD forest loss
  alerts. It is used both to create a dataset for annotation, and during weekly
  inference runs to get the most recent windows to apply the model on.
- Publication: the `rslp.forest_loss_driver.webapp` module publishes the model outputs
  to https://forest-loss.allen.ai. It creates both a single GeoJSON file containing the
  latest predictions, along with a vector tile layer that is used for the Leaflet.js
  map.
- Integrated pipeline: in `rslp/forest_loss_driver/__init__.py` it combines both of the
  above pipelines, along with a call to run the model on the extracted rslearn dataset,
  into one integrated pipeline.
- Dataset configuration file: `data/forest_loss_driver/config.json`.
- Model configuration file: `data/forest_loss_driver/config.yaml`.

The extracted rslearn dataset contains one window per selected GLAD forest loss alert.
It is populated with 6-7 Sentinel-2 images before and after each event, and the
`select_least_cloudy_images` pipeline picks the 3 least cloudy before/after images.

The model classifies the driver in each window (forest loss alert).

## Inference Pipeline Setup

### Environment Variables

Several environment variables are required:
- RSLP_PREFIX: GCS bucket prefix for model checkpoints (`gs://rslearn-eai`).
- `PL_API_KEY`: Planet API key, supplied by `.github/workflows/forest_loss_driver_prediction.yaml`
  from a Github secret. It is no longer used since NICFI is deprecated; instead, now
  the Planet layers are always empty.

There may be others needed that are supplied by `rslp/utils/beaker.py` but not
documented here.

### Configuration

The inference pipeline configuration is at `rslp/forest_loss_driver/config/forest_loss_driver_predict_pipeline_config.yaml`.

- `index_cache_dir` fills in the placeholder in the dataset configuration file. We use
  a temporary directory here since this directory should NOT be shared across inference
  runs, as the available Sentinel-2 scenes will change between them and we want to use
  the latest Sentinel-2 scenes. A stale `index_cache_dir` may mean that the newer
  scenes are not discovered.
- `tile_store_dir`: this is also a placeholder in the dataset config. We keep the tile
  store on WEKA since it stores items (Sentinel-2 scenes) and these scenes should be
  mostly immutable.
- countries and gcs_tiff_filenames: these control which GLAD forest loss alerts we turn
  into rslearn windows. Currently the pipeline runs in Peru only.

### Running the Pipeline

Run the pipeline locally:

```
python -m rslp.main forest_loss_driver integrated_pipeline --integrated_config rslp/forest_loss_driver/config/forest_loss_driver_predict_pipeline_config.yaml
```

The pipeline is set up to run weekly via a Github Action, see
`.github/workflows/forest_loss_driver_prediction.yaml` for details.

## Adding Examples to ES Studio

Here are the steps for adding forest loss driver classification tasks in Brazil and
Colombia to ES Studio.

First, run the alert extraction pipeline:

```
python -m rslp.main forest_loss_driver extract_alerts --ds_path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --extract_alerts_args.gcs_tiff_filenames '["080W_00N_070W_10N.tif", "080W_10S_070W_00N.tif", "070W_10S_060W_00N.tif", "070W_00N_060W_10N.tif"]' --extract_alerts_args.countries '["CO"]' --extract_alerts_args.tile_store_dir "file:///weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/tile_store_root_dir/" --extract_alerts_args.index_cache_dir "file:///tmp/index_cache_dir/" --extract_alerts_args.workers 128 --extract_alerts_args.max_number_of_events 5000 --extract_alerts_args.group 20250428_colombia --extract_alerts_args.days 1095 --extract_alerts_args.prediction_utc_time "2025-03-01 00:00:00+00:00"
python -m rslp.main forest_loss_driver extract_alerts --ds_path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --extract_alerts_args.gcs_tiff_filenames '["050W_20S_040W_10S.tif", "060W_20S_050W_10S.tif", "070W_20S_060W_10S.tif", "040W_10S_030W_00N.tif", "050W_10S_040W_00N.tif", "060W_10S_050W_00N.tif", "070W_10S_060W_00N.tif", "080W_10S_070W_00N.tif", "050W_00N_040W_10N.tif", "060W_00N_050W_10N.tif", "070W_00N_060W_10N.tif", "080W_00N_070W_10N.tif"]' --extract_alerts_args.countries '["BR"]' --extract_alerts_args.tile_store_dir "file:///weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/tile_store_root_dir/" --extract_alerts_args.index_cache_dir "file:///tmp/index_cache_dir/" --extract_alerts_args.workers 128 --extract_alerts_args.max_number_of_events 5000 --extract_alerts_args.group 20250428_brazil --extract_alerts_args.days 1095 --extract_alerts_args.prediction_utc_time "2025-03-01 00:00:00+00:00
```

Switch the rslearn dataset configuration file with the one in
`data/forest_loss_driver/config_rgb_geotiff.json`. This obtains an 8-bit RGB GeoTIFF
for the Sentinel-2 data, along with Planet Labs imagery. Then run the standard prepare,
ingest, and materialize steps.

Use the script `rslp/forest_loss_driver/scripts/populate_label_layer.py` to populate a
placeholder label layer that contains the forest loss polygons. This isn't generated
automatically by the alert extraction pipeline since that is primarily designed to
create a dataset for inference.

Now import into ES Studio:

```
python tools/rslearn_import.py --dataset-path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --layers best_pre_0 best_pre_1 best_pre_2 best_post_0 best_post_1 best_post_2 label planet_monthly --api-url https://earth-system-studio.allen.ai --project-name 'Forest Loss Driver Brazil 7' --always-upload-rasters --workers 64 --groups 20250428_brazil_phase1
python tools/rslearn_import.py --dataset-path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --layers best_pre_0 best_pre_1 best_pre_2 best_post_0 best_post_1 best_post_2 label planet_monthly --api-url https://earth-system-studio.allen.ai --project-name 'Forest Loss Driver Colombia 7' --always-upload-rasters --workers 64 --groups 20250428_colombia_phase1
```
