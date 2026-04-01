This project is for populating examples for new phase of Peru annotation.

## Get Predictions

First we get predictions in Peru for a five-year period. `integrated_config.yaml`
contains the YAML config used for the integrated inference pipeline in
olmoearth_projects:

```
python -m olmoearth_projects.main projects.forest_loss_driver.deploy integrated_pipeline --config ../rslearn_projects/rslp/forest_loss_driver/scripts/peru_20260112/integrated_config.yaml
```

We only need to run it up till it collects the events across the Studio jobs, we got
this file:

```
/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/inference/dataset_20260109/events_from_studio_jobs.geojson
```

## Select Examples

Then we select examples for annotation:

```
python rslp/forest_loss_driver/scripts/peru_20260112/select_examples_for_annotation.py
```

This script will read the events from the file above and write out an rslearn dataset
here:

```
/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/rslearn_dataset_for_selected_events/
```

The rslearn dataset should be first created with config file from
`data/forest_loss_driver/config_studio_annotation.json`.

The selection is done by randomly sampling 100 forest loss events that were predicted
as each of logging/burned/none/river/airstrip (500 total), and another 500 where the
maximum probability is <0.5 (indicating the model was not confident).

## Prepare and Materialize

Make sure to set PLANET_API_KEY env var since it is used in the dataset config. Then:

```
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/rslearn_dataset_for_selected_events/ --workers 128 --retry-max-attempts 10 --retry-backoff-seconds 5
rslearn dataset materialize --root /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/rslearn_dataset_for_selected_events/ --workers 128 --retry-max-attempts 10 --retry-backoff-seconds 5 --ignore-errors
```

## Additional Steps

Afterwards there are a few additional steps we need to do because we forgot to include
it in the initial example selection script.

First, rename the tasks so they have the format `[#113] 2024-05-13 at -8.9846, -76.7046 prediction:burned`:

```
python rslp/forest_loss_driver/scripts/peru_20260112/rename_tasks.py
```

Then, add the label layer (forest loss polygon):

```
python rslp/forest_loss_driver/scripts/peru_20260112/add_label.py
```

## Sync to Studio

Copy to GCS:

```
gsutil -m rsync -r /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/rslearn_dataset_for_selected_events/ gs://ai2-rslearn-projects-data/datasets/forest_loss_driver/dataset_v1/peru_20260112/rslearn_dataset_for_selected_events/
```

Then make request to have it import the dataset (need to create project in Studio first):

```
curl https://olmoearth.allenai.org/api/v1/datasets/ingest --request PUT --header 'Content-Type: application/json' --header "Authorization: Bearer $STUDIO_API_TOKEN" --data '{"dataset_path": "gs://ai2-rslearn-projects-data/datasets/forest_loss_driver/dataset_v1/peru_20260112/rslearn_dataset_for_selected_events/", "project_id": "60e16f40-dbe8-4932-af1b-3f762572530d", "layer_source_names": {}, "prediction_layer_names": []}'
```

After the project is populated, copy the annotation metadata fields from another
project (should have Confidence enum with High/Medium/Low and Area number with 0-9999)
and use `../add_area_to_studio_tasks.py` to set the area in hectares for each polygon.

At 2026-01-20 we sent the project to ACA and they are now looking at it, once
annotation is completed we will need to look into retraining the model.
