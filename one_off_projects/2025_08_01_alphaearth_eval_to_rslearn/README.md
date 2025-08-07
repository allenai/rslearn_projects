This is for comparison on the AlphaEarth supplemental evaluation datasets.
https://zenodo.org/records/16585402

The `create_datasets.py` converts them to rslearn dataset format. Then the data needs
to be materialized.

```
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/aster_ged/ --workers 128 --jobs-per-process 16 --retry-max-attempts 10 --retry-backoff-seconds 5 --disabled-layers landsat
rslearn dataset materialize --root /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/aster_ged/ --workers 128 --retry-max-attempts 10 --retry-backoff-seconds 5 --disabled-layers landsat --ignore-errors
```

`rslp.helios.get_embeddings` can be used to compute embeddings with Helios model. Then
we can run `get_balanced_accuracy.py` to compute the balanced accuracy.

```
python -m rslp.helios.get_embeddings --ds_path /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/africa_crop_mask/ --patch_size 1 --checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/favyen_decode_gse_worldcover_osm_srtm_titan/step370000 --input_size 32 --embed_fname helios_embeddings_favyen_decode_gse_worldcover_osm_srtm_titan_ws32_ps1.npy
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/africa_crop_mask/ --repeats 10 --samples 200 --k 3 --classes not_crop,crop --embed_fname helios_embeddings_favyen_decode_gse_worldcover_osm_srtm_titan_ws64_ps4.npy
```

The following datasets have been materialized and can be used for evaluation:

- /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/africa_crop_mask/
- /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/ethiopia_crops/
- /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/lcmap_lu/
- /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/us_trees/

`aster_ged` is materialized but it is regression and the code for evaluation is not implemented yet.
