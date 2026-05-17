Scripts to subset WorldCover in two ways to have two new tasks in olmoearth_evals
(200 per class and 500 per class), and to take AEF datasets and crop them to 32x32
so we can train on them much more efficiently.

- `convert_worldcover.py`: subsets WorldCover.
- `convert_aef_to_32.py`: copies the 8 AEF eval datasets into
  `olmoearth_evals/<task>/`, keeping only Sentinel-2 raster groups, the
  `label_raster` raster layer, and the `label` vector layer, with every kept
  GeoTIFF cropped to a centered 32x32 region. The
  `data/olmoearth_evals/tasks/{africa_crop_mask,canada_crops_coarse,canada_crops_fine,descals,ethiopia_crops,glance,lcmap_lu,us_trees}.yaml`
  configs already point at the new path.
