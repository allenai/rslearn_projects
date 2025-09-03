Here is example:

```
python -m rslp.main esrun esrun --config_path esrun_data/satlas/solar_farm/ --scratch_path /tmp/scratch/
```

So in `esrun_data/satlas/solar_farm/` we have:

- `dataset.json`: the rslearn dataset configuration file.
- `model.yaml`: the rslearn model configuration file.
- `esrun.yaml`: new YAML file containing esrun pre/post processing config.
- `prediction_request_geometry.geojson`: the GeoJSON input to the esrun partition and window generation.
