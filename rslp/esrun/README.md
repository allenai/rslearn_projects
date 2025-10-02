Here is example:

```
python -m rslp.main esrun esrun --config_path esrun_data/satlas/solar_farm/ --scratch_path /tmp/scratch/
```

So in `esrun_data/satlas/solar_farm/` we have:

- `dataset.json`: the rslearn dataset configuration file.
- `model.yaml`: the rslearn model configuration file.
- `esrun.yaml`: new YAML file containing esrun pre/post processing config.
- `prediction_request_geometry.geojson`: the GeoJSON input to the esrun partition and window generation.


In the `esrun_data/sample` directory, we can also run training window preparation, which
depends on:

- `dataset.json`: the rslearn dataset configuration file.
- `esrun.ymal`: new YAML file containiner the window_prep config
- `annotation_features.geojson`: annotation geojson FeatureCollection exported from Studio
- `annotation_task_features.geojson`: the Studio task geojson Features corresponding to the above

Run with:

```
python -m rslp.main esrun prepare_labeled_windows \
    --project_path esrun_data/sample \
    --scratch_path /tmp/scratch
```

to produce a new dataset at:

```
/tmp/scratch/dataset
```
