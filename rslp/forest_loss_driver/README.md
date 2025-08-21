# Forest Loss Driver

This project aims to develop a model to classify the driver (e.g. mining,
agriculture, road, storm, landslide, etc.) of detected forest loss. Currently the focus
is in the Amazon basis, specifically in Peru, Brazil, and Colombia. Currently we use
GLAD forest loss alerts as the forest loss detection system. We find connected
components of GLAD alert pixels, threshold the area, and then apply the model on each
polygon. The model inputs 3-5 images before the detected forest loss and 3-5 images
after the detected forest loss, and classifies the driver.

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
`data/forest_loss_driver/config_studio_annotation.json`. This obtains an 8-bit RGB
GeoTIFF for the Sentinel-2 data, along with Planet Labs imagery. Then run the standard
prepare, ingest, and materialize steps.

Use the script `rslp/forest_loss_driver/scripts/populate_label_layer.py` to populate a
placeholder label layer that contains the forest loss polygons. This isn't generated
automatically by the alert extraction pipeline since that is primarily designed to
create a dataset for inference.

We also had a script `rslp/forest_loss_driver/scripts/select_windows_in_amazon.py`
since ACA requested to focus on Amazon and not other areas of the countries (they
provided a bounding polygon to apply). The script moves the windows outside that
polygon to other groups in the rslearn dataset.

Now import into ES Studio:

```
python tools/rslearn_import.py --dataset-path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --layers best_pre_0 best_pre_1 best_pre_2 best_post_0 best_post_1 best_post_2 label planet_monthly --api-url https://earth-system-studio.allen.ai --project-name 'Forest Loss Driver Brazil 7' --always-upload-rasters --workers 64 --groups 20250428_brazil_phase1
python tools/rslearn_import.py --dataset-path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --layers best_pre_0 best_pre_1 best_pre_2 best_post_0 best_post_1 best_post_2 label planet_monthly --api-url https://earth-system-studio.allen.ai --project-name 'Forest Loss Driver Colombia 7' --always-upload-rasters --workers 64 --groups 20250428_colombia_phase1
```

The script `rslp/forest_loss_driver/scripts/add_area_to_studio_tasks.py` can then be
run (after configuring basemaps and annotation tags) to have the area of the polygon
show up as a metadata value.

## Select Additional Examples to Label

For the Peru+Brazil+Colombia project (2025), we first picked 500 examples to label from
the "Adding Examples" workflow above. We use that to prioritize what else to label by
training a model on Peru examples + Brazil/Colombia, and then look at the output
classes and probabilities.

First we sync the labels from Studio. ACA proposed an updated label hierarchy for this
project, but here we are mapping it back to our old hierarchy for compatibility with
the Peru labels.

```
python -m rslp.forest_loss_driver.scripts.sync_labels_from_studio --project_id f56e41c6-83ab-4a7f-9b14-443391f9b2ba --ds_path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --remap_labels
python -m rslp.forest_loss_driver.scripts.sync_labels_from_studio --project_id a493cba0-466f-4604-8359-c437b78f7009 --ds_path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --remap_labels
```

Create a combined dataset with the Peru labels. We use the multimodal config but really we just use Sentinel-2 images here.

```
mkdir /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined/
cp data/forest_loss_driver/config_multimodal.json /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined/config.json
rsync -av /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/windows/{20250428_brazil_phase1,20250428_colombia_phase1} /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined/windows/
rsync -av /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/20250605/windows/* /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined/windows/
# Re-materialize since the source dataset may have had different config.
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined/ --workers 64 --disabled-layers pre_landsat,post_landsat,pre_sentinel1,post_sentinel1 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined/ --workers 64 --disabled-layers pre_landsat,post_landsat,pre_sentinel1,post_sentinel1 --ignore-errors --retry-max-attempts 5 --retry-backoff-seconds 5
# Assign split.
python -m rslp.forest_loss_driver.scripts.assign_split /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined/
```

Next we can train a model on the data.

```
python -m rslp.rslearn_main model fit --config data/forest_loss_driver/model_for_phase2/config_helios_frozen.yaml
```

This model performs well but all of these configs have been updated for training on
this data.

- `data/forest_loss_driver/model_for_phase2/config_satlaspretrain.yaml`
- `data/forest_loss_driver/model_for_phase2/config_helios.yaml`
- `data/forest_loss_driver/model_for_phase2/config_helios_frozen.yaml`

Apply the model to the remaining collected windows:

```
python -m rslp.rslearn_main model predict --config data/forest_loss_driver/config_helios_frozen.yaml --data.init_args.path=/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --data.init_args.predict_config.groups=["20250428_colombia","20250428_brazil"] --load_best=true
```

Then we have a script to select windows:

- Select 50 each from Brazil/Colombia predicted as road/logging/mining/river/landslide (500 total)
- Select 150 each from Brazil/Colombia predicted not as the above classes with max(probabilities) < 0.6 (300 total)

```
python -m rslp.forest_loss_driver.scripts.select_for_phase2
```

After doing this I was worried that the selected annotations for phase 2 might be too
close to the initial set of annotations, since the `select_for_phase2.py` will create
grid index to avoid picking annotations within 500 m of each other but only within the
new set, not comparing against the previous set. But I wrote a quick script to check
the selections against the previous set and it looks like only 4 are within 500 m of a
phase 1 annotation (14 within 1 km, 75 within 5 km).

Also I realized I actually intended to ensure the new annotations are at least 500
pixels (5 km) from each other, not 500 m, but I checked this too and 197 of the 682
total have another annotation within 5 km. I think that is okay, most of them are
fairly isolated, and 50 pixels away is still something.

Once these examples are annotated we should train the model again, but focus more on
accuracy instead of using it to prioritize what else to annotate.

## Train Model for Brazil and Colombia

Download and extract the dataset from here:

- TODO

Now train the model:

```
python -m rslp.rslearn_main model fit --config data/forest_loss_driver/brazil_colombia_model/config_satlaspretrain.yaml
```

It should show loss, accuracy, confusion matrix, etc. on W&B.

Get outputs from the model:

```
python -m rslp.rslearn_main model predict --config data/forest_loss_driver/brazil_colombia_model/config_satlaspretrain.yaml --data.init_args.predict_config.groups='["20250428_brazil_phase1","20250428_colombia_phase1"]'
```
