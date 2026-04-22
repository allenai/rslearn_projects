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

### Next step (not yet implemented)

The annotated months will drive creation of an rslearn dataset of
time series intersecting the change period, used to train a supervised model
that detects change closer to when it actually occurs (the current
segmentation-based approach only picks up change ~3 years after it happens).
