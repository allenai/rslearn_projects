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
- `convert_vessel_attribute.py`: builds a small subset of the
  `sentinel2_vessel_attribute` dataset. Only windows with both `type` and
  `length` are kept. Train split is sampled as the union of 500 per vessel type
  and 500 per length bucket ([0,50), [50,75), …, [225,250), [250,+inf)); val
  and test are copied in full. All sentinel2 images are center-cropped from
  128x128 to 64x64. Output goes to
  `olmoearth_evals/small_sentinel2_vessel_attribute/`. Task configs:
  `data/olmoearth_evals/tasks/small_sentinel2_vessel_{type,length}.yaml`.
- `convert_mangrove.py`: subsets mangrove classification (20250626) to 2000
  Mangrove + 2000 non-Mangrove for train; val and test copied in full. No
  cropping needed (already 32x32). Output goes to `olmoearth_evals/small_mangrove/`.
  Task configs: `data/olmoearth_evals/tasks/small_mangrove_{base,mm}.yaml`.
- `convert_eurosat.py`: subsets EuroSAT (27K windows, 10 classes) to 200 per
  class per split (train/val/test = 2000/2000/2000). No cropping needed
  (already 64x64). Output goes to `olmoearth_evals/small_eurosat/`.
  Task config: `data/olmoearth_evals/tasks/small_eurosat.yaml`.
