# Landsat vessel scripts

Run from the repo root as `python -m rslp.landsat_vessels.scripts.<name>`; each script's
docstring has its full usage.

| script | what it does |
|---|---|
| `create_round1_windows.py` | Create one 512 px @ 15 m window per detection in an annotation pool, with the source scene pinned and frozen scene-level splits. Input to the annotation app. |
| `render_annotation_assets.py` | Render the crop / context images and spectral curves the annotation app (`../annotations/`) serves. |
| `visualize_predictions.py` | Visualize the outputs of `model predict`. |
| `sample_request.py` | Send a sample request to a running Landsat vessels API (`api_main.py`). |
| `sync_gcs_to_weka.yaml` | Beaker experiment spec that syncs the Landsat scene zips and model checkpoints from GCS to Weka. |
