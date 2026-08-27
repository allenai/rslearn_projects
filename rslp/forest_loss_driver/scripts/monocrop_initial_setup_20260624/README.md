# Monocrop Initial Setup (2026-06-24)

A new annotation round for forest loss driver, focused on **monocultures** (oil palm,
soybean) in Peru, Bolivia, and Ecuador, using longer-term imagery (2022-2025). This
directory contains the scripts to go from raw GLAD forest loss alerts to per-AOI
agriculture annotation sets ready to import into ES Studio.

The classification model and dataset extraction pipeline live in
`olmoearth_projects.projects.forest_loss_driver`; these scripts wrap and post-process
that pipeline for this specific round.

The original run wrote everything under:

```
/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/20260622_monocrop_setup/predictions_2022_to_2025/
```

## Pipeline overview

```
GLAD alerts ──(1) extract_alerts (olmoearth_projects)──► prediction_request_geometry.geojson
        │  (2) make_batches
        ▼
 ..._batch{i}_orig.geojson   (polygons, kept for later)
 ..._batch{i}_simple.geojson (centroids, uploaded to Studio)
        │  (3) Studio inference (latest forest loss driver model)
        ▼
 result_batch{i}.geojson     (points + new_label)
        │  (4) make_annotation_sets
        ▼
 {AOI}_agriculture.geojson, {AOI}_agriculture_large.geojson
        │  (upload to Studio) (5) rename_studio_tasks
        ▼
 [#001] (-5.7671, -77.1391) at 2022-01-08   (renamed Studio tasks)
```

## 1. Extract forest loss events

Extract connected-component forest loss events from GLAD alerts for **Peru (PE),
Bolivia (BO), and Ecuador (EC)**, from **2022-01-01 through 2025-12-31**, sampling in
**90-day slices** (`slice_days=90`). Slicing keeps connected components from different
time periods from merging together, since the GLAD date raster stores a single date per
pixel.

This step calls the `extract_alerts` workflow in `olmoearth_projects` directly (there is
no wrapper script). The relevant GLAD 10°×10° alert tiles for these countries are:

| Tile | Rough coverage |
| --- | --- |
| `080W_00N_070W_10N.tif` | northern Ecuador |
| `080W_10S_070W_00N.tif` | southern Ecuador / Peru |
| `080W_20S_070W_10S.tif` | central/southern Peru |
| `070W_10S_060W_00N.tif` | eastern Peru / northern Bolivia |
| `070W_20S_060W_10S.tif` | central Bolivia (most soybean events) |
| `060W_20S_050W_10S.tif` | eastern Bolivia |

A tile name encodes its lon/lat corners (e.g. `080W_20S_070W_10S` spans lon −80..−70,
lat −20..−10). The country filter clips events to the actual country polygons, so
over-including a tile is harmless; add neighbors to widen coverage. Note that GLAD
coverage stops at lat −20°, so there is no tile for the small sliver of the Bolivia AOI
below −20°.

Run from the **olmoearth_projects** repo root, in its environment. The window is the
`days` preceding `prediction_utc_time`; we anchor at 2026-01-01 and look back 1461 days
to cover 2022-01-01 onward:

```bash
python -m olmoearth_projects.main projects.forest_loss_driver extract_alerts \
    --extract_alerts_args.gcs_tiff_filenames+=080W_00N_070W_10N.tif \
    --extract_alerts_args.gcs_tiff_filenames+=080W_10S_070W_00N.tif \
    --extract_alerts_args.gcs_tiff_filenames+=080W_20S_070W_10S.tif \
    --extract_alerts_args.gcs_tiff_filenames+=070W_10S_060W_00N.tif \
    --extract_alerts_args.gcs_tiff_filenames+=070W_20S_060W_10S.tif \
    --extract_alerts_args.gcs_tiff_filenames+=060W_20S_050W_10S.tif \
    --extract_alerts_args.countries+=PE \
    --extract_alerts_args.countries+=BO \
    --extract_alerts_args.countries+=EC \
    --extract_alerts_args.prediction_utc_time=2026-01-01T00:00:00+00:00 \
    --extract_alerts_args.days=1461 \
    --extract_alerts_args.slice_days=90 \
    --extract_alerts_args.smooth_polygons=false \
    --extract_alerts_args.workers=64 \
    --extract_alerts_args.country_data_path=/weka/dfive-default/rslearn-eai/artifacts/natural_earth_countries/20240830/ne_10m_admin_0_countries.shp \
    --extract_alerts_args.out_fname=/weka/.../20260622_monocrop_setup/predictions_2022_to_2025/prediction_request_geometry.geojson
```

- `country_data_path` is the Natural Earth 10m admin-0 countries shapefile
  ([download](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/)),
  with its auxiliary `.dbf`/`.shx`/etc. files alongside.
- `max_number_of_events` is the per-slice, per-tile cap; **omit it (default `None`) to
  keep every connected component**, which is required if you later need to match these
  components against a previous run. To cap instead, add
  `--extract_alerts_args.max_number_of_events=2000` (the first run used `2000`,
  producing ~130k events total).
- `smooth_polygons=false` disables the 1-pixel buffer + simplify, keeping the raw,
  blocky pixel-edge outline of each event. Drop it (default `true`) to get the rounded /
  simplified shapes.

Output: a single `prediction_request_geometry.geojson` with one **polygon** feature per
event and properties `center_pixel`, `tif_fname`, `oe_start_time`, `oe_end_time`,
`country`.

## 2. Split into batches and simplify

Studio inference jobs are capped at 10000 events, so split the geometry into batches and
create the simplified centroid copies that get uploaded to Studio. For each batch this
writes `..._batch{i}_orig.geojson` (polygons, kept so we can restore them after
inference) and `..._batch{i}_simple.geojson` (centroid Points, uploaded to Studio —
Points avoid job failures from complex polygon geometries).

Run in an environment with `shapely` (e.g. the rslearn venv):

```bash
python rslp/forest_loss_driver/scripts/monocrop_initial_setup_20260624/make_batches.py \
    --input /weka/.../predictions_2022_to_2025/prediction_request_geometry.geojson \
    --output-dir /weka/.../predictions_2022_to_2025/
```

The original run produced 13 batches of 10000 (`batch0`..`batch12`).

## 3. Run prediction in Studio

Upload each `..._batch{i}_simple.geojson` to ES Studio and run inference with the
**latest forest loss driver classification model**, then download each job's result.
Lay the downloads out so each batch's result is at:

```
<predictions_dir>/result_batch{i}.geojson
```

Each `result_batch{i}.geojson` has **point** geometries (the submitted centroids) plus a
`new_label` property (the predicted driver, e.g. `agriculture`, `mining`, `burned`, …)
and per-class `probs`.

> The deploy pipeline normally drives Studio inference programmatically (see
> `olmoearth_projects.projects.forest_loss_driver.deploy`); for this round the jobs were
> run and downloaded manually.

## 4. Build per-AOI agriculture annotation sets

This consolidated step takes the Studio predictions, the original polygon batches, and
the AOI definitions, and produces the final annotation sets. It:

1. **Restores polygons**: matches each Studio result point back to its original polygon
   by nearest centroid (an exact match, since simplification used the centroid) and
   copies `new_label`/`probs` onto the polygon. If `--regen` is given (a non-smoothed
   `prediction_request_geometry.geojson`, e.g. from re-running step 1 with
   `smooth_polygons=false`), each event's polygon geometry is replaced by the matching
   one from that file (matched by `(tif_fname, center_pixel)`); unmatched events keep
   their smoothed geometry. All other attributes still come from the per-batch orig
   files.
2. **Filters per AOI**: keeps agriculture polygons whose centroid is inside each AOI and
   writes, per AOI:
   - `{AOI}_agriculture.geojson` — up to `--sample-agriculture` (default 2000) polygons.
   - `{AOI}_agriculture_large.geojson` — up to `--sample-large` (default 1000) polygons
     whose area exceeds `--min-hectares` (default 5 ha). This set is drawn
     independently and may overlap `{AOI}_agriculture.geojson`. Area is computed by
     reprojecting each polygon to its local UTM/UPS projection with a 100 m pixel size,
     so the projected area equals hectares directly (same method as
     `../add_area_to_studio_tasks.py`).

AOIs are polygon files (one per AOI), passed as `NAME=path`. Both standard GeoJSON
(Polygon/MultiPolygon) and Esri JSON (`rings`) are accepted; multiple features in one
file are unioned.

Run in an environment with `rslearn`, `shapely`, `scipy`, `numpy` (e.g. the rslearn venv):

```bash
python rslp/forest_loss_driver/scripts/monocrop_initial_setup_20260624/make_annotation_sets.py \
    --predictions-dir /path/to/studio_results \
    --orig-dir /weka/.../predictions_2022_to_2025 \
    --aoi Oilpalmecuador=/path/Oilpalmecuador.json \
    --aoi Oilpalmperu=/path/Oilpalmperu.json \
    --aoi Soybeanbolivia=/path/Soybeanbolivia.json \
    --output-dir /path/to/output \
    --write-merged
```

- `--write-merged` additionally writes per-batch `merged_batch{i}.geojson` (all polygons
  with labels), useful for inspection or other slicing.
- `--target-label` defaults to `agriculture`; pass a comma-separated list to keep
  multiple classes.
- `--regen` (optional) replaces each event's polygon geometry with the non-smoothed one
  from the given `prediction_request_geometry.geojson` (matched by
  `(tif_fname, center_pixel)`); unmatched events keep the smoothed geometry. Use this
  with the `predictions_2022_to_2025_regen/` output to get raw, blocky pixel-edge
  polygons while keeping all labels/attributes from the original batches.
- `--workers` (default 64) parallelizes the per-AOI centroid matching and the area
  computation.

For the original run this produced (oil palm Ecuador / oil palm Peru / soybean Bolivia):

| AOI | `_agriculture` | `_agriculture_large` (>5 ha) |
| --- | --- | --- |
| Oilpalmecuador | 2000 | 320 |
| Oilpalmperu | 2000 | 460 |
| Soybeanbolivia | 110 | 0 |

> Note: Soybeanbolivia had only 110 agriculture events in its AOI, and very few exceed
> 5 ha, so `_agriculture_large` is small or empty. Only ~5–6% of agriculture polygons
> exceed 5 ha, since forest-loss patches are mostly small. `_agriculture_large` is drawn
> independently from `_agriculture` and the two may overlap.

## 5. Rename Studio tasks

After uploading an annotation set to a Studio project, the tasks are named after their
upload-time GeoJSON (e.g. `Oilpalmperu_agriculture_large_-5.767112_-77.139067`). This
step renames every task in the project to a compact, shuffled, counter-prefixed name:

```
Oilpalmperu_agriculture_large_-5.767112_-77.139067
    →  [#001] (-5.7671, -77.1391) at 2022-01-08
```

- `#001` is a 1-based counter over a **random shuffle** of the project's tasks
  (zero-padded to 3 digits; `--seed` controls the shuffle).
- `(-5.7671, -77.1391)` is `(lat, lon)` parsed from the trailing two underscore-separated
  floats of the original name, rounded to 4 decimals.
- `2022-01-08` is the task's `start_time` (date only).

The script skips tasks that are already renamed (name starts with `[#`) and continues
the counter after the highest existing `[#NNN]`, so it is safe to re-run after uploading
more tasks to the same project. Set `STUDIO_API_KEY` and run in any environment with
`requests` and `tqdm` (e.g. the rslearn venv):

```bash
STUDIO_API_KEY=... python \
    rslp/forest_loss_driver/scripts/monocrop_initial_setup_20260624/rename_studio_tasks.py \
    --project-id <PROJECT_ID> \
    --dry-run
```

- `--dry-run` prints the planned `old -> new` renames without calling the API. Drop it to
  apply the renames.
- `--seed` (default 42) controls the shuffle that assigns counters.

## Files

- [make_batches.py](make_batches.py) — step 2: split into batches of 10000 and write
  the `_orig` (polygon) and `_simple` (centroid) GeoJSONs.
- [make_annotation_sets.py](make_annotation_sets.py) — step 4: merge predictions back
  onto polygons (optionally swapping in non-smoothed `--regen` geometry) and produce the
  per-AOI `_agriculture` and `_agriculture_large` sets.
- [rename_studio_tasks.py](rename_studio_tasks.py) — step 5: rename uploaded Studio tasks
  to `[#NNN] (lat, lon) at DATE` over a random shuffle.
