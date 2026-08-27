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
    --checkpoint_path /weka/dfive-default/rslearn-eai/projects/2026_04_05_worldcover_change/olmoearth_base_s2_6mo_ws32_ps4_00/best.ckpt
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

```python
import json
import random
from collections import defaultdict

IN_PATH = "/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/land_cover_change_src_dst.geojson"
OUT_PATH = "/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/land_cover_change_src_dst_sel100.geojson"
PER_PAIR = 100

with open(IN_PATH) as f:
    fc = json.load(f)

change_by_pair = defaultdict(list)
no_change_by_window = {}
for feat in fc["features"]:
    p = feat["properties"]
    if p["feature_type"] == "no_change":
        no_change_by_window[(p["window_group"], p["window_name"])] = feat
    elif p["feature_type"] == "change":
        change_by_pair[(p["src_class_name"], p["dst_class_name"])].append(feat)

selected_change = []
for pair, feats in change_by_pair.items():
    selected_change.extend(random.sample(feats, min(PER_PAIR, len(feats))))

selected_windows = {
    (f["properties"]["window_group"], f["properties"]["window_name"])
    for f in selected_change
}
selected_no_change = [
    no_change_by_window[wkey] for wkey in selected_windows if wkey in no_change_by_window
]

out_fc = {"type": "FeatureCollection", "features": selected_change + selected_no_change}
with open(OUT_PATH, "w") as f:
    json.dump(out_fc, f)

print(f"Wrote {len(selected_change)} change + {len(selected_no_change)} no-change features to {OUT_PATH}")
```

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
- Annotation form, positioned right above the Sentinel-2 panels so the
  imagery stays visible while labeling. The four `YYYY-MM` fields are:
  - `pre_change`: a month where the model would still predict the source
    category.
  - `change_start`: the first image where the change is visible.
  - `change_end`: the last image where the change is still visible.
  - `post_change`: a month where the model would predict the destination
    category.
- Sentinel-2 panels for the selected year, plus the early and late land
  cover class maps (band 2 and band 3 of `land_cover_change.tif`).
- `Overlay` (hotkey `a`) toggles the change + no-change masks on top of
  every tile.
- Press **Save** to persist the four fields. The server keys the entry
  by its `feature_idx` (the feature's index in the source GeoJSON) and
  writes only the annotations file, which is much faster than rewriting
  the entire GeoJSON on every save.

Annotations are stored in a sidecar JSON file next to the GeoJSON,
defaulting to `<geojson_stem>.annotations.json` (pass `--annotations
<path>` to override). The file is a list of dicts:

```json
[
  {
    "feature_idx": 12,
    "window_group": "...",
    "window_name": "...",
    "src_class_name": "tree",
    "dst_class_name": "crops",
    "pre_change": "2017-06",
    "change_start": "2018-04",
    "change_end": "2019-09",
    "post_change": "2020-08"
  }
]
```

On first load, if no annotations file exists yet, the server migrates
any legacy `change_start_month`/`change_end_month` props from the
GeoJSON into `change_start`/`change_end` (in-memory only; the GeoJSON
itself is never rewritten).

Keyboard shortcuts: `p`/`n` previous/next example, `←`/`→` previous/next
year, `a` toggle overlay.

### 4. Train per-event change detector (`land_cover_time_series_change_model/`)

The annotated months drive creation of an rslearn dataset of time series
intersecting the change period, used to train a supervised model that detects
change closer to when it actually occurs (the current segmentation-based
approach only picks up change ~3 years after it happens).

Model setup:

- Input: 12 quarterly Sentinel-2 mosaics (3 years).
- Output (multi-task per pixel):
  - `binary` — `nodata` / `no_change` / `change`
  - `src`    — source WorldCover class at changed pixels
  - `dst`    — destination WorldCover class at changed pixels

Dataset windows are built with 20 quarterly mosaics over 5 years. At training
time, `TimeSeriesChangeSubsample` (in
`land_cover_time_series_change_model/transforms.py`) picks a 12-mosaic slice
that brackets at least one side of the annotated change event. The src and dst
targets are masked when the chosen slice doesn't bracket the corresponding
transition, so each head is only trained on samples where its label is
meaningful.

#### Create the dataset

```bash
SRC=/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408
OUT=/weka/dfive-default/rslearn-eai/datasets/change_finder/ts_change_v1

cp data/change_finder/land_cover_time_series_change_model/config.json "$OUT/config.json"

python -m rslp.change_finder.land_cover_time_series_change_model.create_windows \
    --polygons_geojson "$SRC/land_cover_change_src_dst_sel100.geojson" \
    --annotations_json "$SRC/land_cover_change_src_dst_sel100.annotations.json" \
    --src_ds_path "$SRC" \
    --out_ds_path "$OUT"
```

This writes:

- new rslearn windows (one per annotated change feature), each with the 5-year
  time range and the four annotated months stored in `window.options`
- pre-rasterized `label_binary`, `label_src`, `label_dst` GeoTIFFs per window
  (with `mark_layer_completed` so rslearn skips them at materialize time)
- a sidecar `ts_change_annotations.json` at the dataset root used by the
  training-time multi-task wrapper to look up annotations

#### Materialize the Sentinel-2 mosaics

```bash
rslearn dataset prepare     --root "$OUT" --workers 32
rslearn dataset materialize --root "$OUT" --workers 32
```

#### Train

```bash
rslearn model fit --config data/change_finder/land_cover_time_series_change_model/config.yaml
```

#### Predict over an AOI

Create a prediction dataset with 1024x1024 UTM windows at 10 m/pix. Use a
3-year time range so rslearn materializes exactly 12 quarterly Sentinel-2
mosaics (matching the trained input length).

```bash
PREDICT_DS=/weka/dfive-default/rslearn-eai/datasets/change_finder/ts_change_predict_<aoi>
mkdir -p "$PREDICT_DS"
cp data/change_finder/land_cover_time_series_change_model/config.json "$PREDICT_DS/config.json"

rslearn dataset add_windows \
    --root "$PREDICT_DS" \
    --group predict \
    --fname aoi.geojson \
    --src_crs EPSG:4326 \
    --utm --resolution 10 --grid_size 1024 \
    --start 2022-01-01T00:00:00+00:00 \
    --end 2025-01-01T00:00:00+00:00

rslearn dataset prepare     --root "$PREDICT_DS" --workers 32
rslearn dataset ingest      --root "$PREDICT_DS" --workers 32
rslearn dataset materialize --root "$PREDICT_DS" --workers 32

rslearn model predict --config data/change_finder/land_cover_time_series_change_model/config.yaml --data.init_args.path="$PREDICT_DS"
```

Output: one 29-band uint8 GeoTIFF per prediction window at
`<window>/layers/output_change/<bandset>/geotiff.tif`. Probabilities are
scaled 0-255. Band layout:

- 0..2: binary probs (nodata / no\_change / change)
- 3..15: src class probs (13 WorldCover classes, index 0 = nodata)
- 16..28: dst class probs (13 WorldCover classes, same order)
