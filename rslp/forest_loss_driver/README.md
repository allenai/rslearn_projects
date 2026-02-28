# Forest Loss Driver

This project aims to develop a model to classify the driver (e.g. mining,
agriculture, road, storm, landslide, etc.) of detected forest loss. Currently the focus
is in the Amazon basis, specifically in Peru, Brazil, and Colombia. Currently we use
GLAD forest loss alerts as the forest loss detection system. We find connected
components of GLAD alert pixels, threshold the area, and then apply the model on each
polygon. The model inputs 3-5 images before the detected forest loss and 3-5 images
after the detected forest loss, and classifies the driver.

The inference pipeline and dataset extraction pipeline have been moved to
olmoearth_projects, see `olmoearth_projects.projects.forest_loss_driver.deploy`.

The rslearn dataset (both for training and for prediction) is derived from GLAD forest
loss alerts published on GCS at `gs://earthenginepartners-hansen/S2alert/`. The dataset
contains one window per selected GLAD forest loss alert. The model inputs several
Sentinel-2 images before each event and several images after, and it classifies the
driver in each window (forest loss alert).

For a summary of the version history of dataset and model configuration files, see
`data/forest_loss_driver/README.md`.

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
rslearn model fit --config data/forest_loss_driver/model_for_phase2/config_helios_frozen.yaml
```

This model performs well but all of these configs have been updated for training on
this data.

- `data/forest_loss_driver/model_for_phase2/config_satlaspretrain.yaml`
- `data/forest_loss_driver/model_for_phase2/config_helios.yaml`
- `data/forest_loss_driver/model_for_phase2/config_helios_frozen.yaml`

Apply the model to the remaining collected windows:

```
rslearn model predict --config data/forest_loss_driver/model_for_phase2/config_helios_frozen.yaml --data.init_args.path=/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/ --data.init_args.predict_config.groups=["20250428_colombia","20250428_brazil"] --load_checkpoint_mode=best
```

Then we have a script to select windows:

- Select 50 each from Brazil/Colombia predicted as road/logging/mining/river/landslide (500 total). These classes
  appeared less frequently in the initial set of 500 examples.
- Select 150 each from Brazil/Colombia predicted not as the above classes with max(probabilities) < 0.6 (300 total).
  This way we are not fully biasing towards what the model more confidently thought was the more rare classes, so we
  may see examples that fall into a rare class but the model expressed low overall confidence.

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

- https://storage.googleapis.com/ai2-rslearn-projects-data/datasets/forest_loss_driver/dataset_v1/combined.tar

Now train the model:

```
rslearn model fit --config data/forest_loss_driver/brazil_colombia_model/config_satlaspretrain.yaml
```

It should show loss, accuracy, confusion matrix, etc. on W&B.

Get outputs from the model:

```
rslearn model predict --config data/forest_loss_driver/brazil_colombia_model/config_satlaspretrain.yaml --data.init_args.predict_config.groups='["20250428_brazil_phase1","20250428_colombia_phase1"]'
```
