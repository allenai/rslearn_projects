## Land Cover Change Pipeline

We use a segmentation model trained on the WorldCover labels to produce per-pixel
land cover predictions on a dataset with ten years of Sentinel-2 images. Then we find
pixels that transition confidently from one class to another between the earliest and
most recent years in the dataset.

- Dataset: `/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/`
- Land cover model config: `data/land_cover_change/worldcover_change/config.yaml`
- Land cover model checkpoint: `/weka/dfive-default/rslearn-eai/projects/2026_04_05_worldcover_change/olmoearth_base_s2_6mo_ws32_ps4_00/best.ckpt`

### 1. Per-window segmentation (`compute_land_cover_change.py`)

Runs a segmentation head on each of the 10 `sentinel2_yN` layers per window
and writes `land_cover_change.tif` inside the window directory. The GeoTIFF
bands are:

- B0: binary change flag (early dominant class != late dominant class)
- B1: early dominant class id
- B2: late dominant class id
- B3..: 13 per-class probability bands per year (uint8, 0-255), stored in
  early-years-then-late-years order

```bash
python -m rslp.change_finder.compute_land_cover_change \
    --ds_path /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/ \
    --checkpoint_path /path/to/segmentation.ckpt
```

### 2. Extract change polygons (`scripts/create_land_cover_change_geojson.py`)

Aggregates the per-window GeoTIFFs into a single GeoJSON. For each window
and each pair of `(src_class, dst_class)` it keeps every pixel where the
src class probability exceeds `--threshold` in every early timestep AND the
dst class probability exceeds `--threshold` in every late timestep.

Output features come in two flavors, distinguished by `properties.feature_type`:

- `"change"` — one feature per `(window, src, dst)` tuple. All connected
  components with `>= min_pixels` are merged into a single `MultiPolygon`.
  Carries `src_class_id`/`src_class_name`/`dst_class_id`/`dst_class_name`
  and a `num_pixels` count.
- `"no_change"` — at most one per window. Union across classes of pixels
  that were confidently the same class across every early *and* late
  timestep (class 0 / nodata excluded). Only emitted for windows that
  already have at least one `change` feature, since these are purely for
  visualization alongside change features. No src/dst properties.

```bash
python -m rslp.change_finder.scripts.create_land_cover_change_geojson \
    --ds_path /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/ \
    --out_path /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/land_cover_change_src_dst.geojson
```

I used a separate script to randomly selected 100 from each (src, dst) pair to create
a new GeoJSON `land_cover_change_src_dst_sel100.geojson`.

### 3. Browse & annotate (`land_cover_change_viewer/`)

Flask UI for scanning through the change features, confirming the change
mask (yellow) and the accompanying no-change mask (blue) look right, and
labeling the approximate time range over which the change happened.

```bash
python -m rslp.change_finder.land_cover_change_viewer.server \
    --geojson /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/land_cover_change_src_dst_sel100.geojson \
    --ds-path /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/ \
    --port 8080
```

The UI:

- Top bar: `From` and `To` dropdowns (each with an `All` option; default is
  `All -> All`, which shows every change feature in a deterministic but
  src/dst-interleaved order).
- Sentinel-2 panels for the selected year, plus the early and late land
  cover class maps (band 2 and band 3 of `land_cover_change.tif`).
- `Overlay` (hotkey `a`) toggles the change + no-change masks on top of
  every tile.
- Bottom annotation form: enter `change_start_month` and `change_end_month`
  as `YYYY-MM` (either can be blank to clear) and press **Save**. The
  server finds the feature by `(window_group, window_name, src_class_id,
  dst_class_id)`, updates its properties, and rewrites the GeoJSON
  atomically via `rslearn.utils.fsspec.open_atomic`. No-change features
  are preserved untouched during the rewrite.

Keyboard shortcuts: `p`/`n` previous/next example, `←`/`→` previous/next
year, `a` toggle overlay.

### Next step (not yet implemented)

The annotated start/end months will drive creation of an rslearn dataset of
time series intersecting the change period, used to train a supervised model
that detects change closer to when it actually occurs (the current
segmentation-based approach only picks up change ~3 years after it happens).
