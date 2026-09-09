# LCC Seg model: land cover transition segmentation

A simpler alternative to the dual-pass `lcc_model`. It detects specific land cover
transitions as a plain per-pixel **segmentation** task, using a single 12-image
Sentinel-2 stack fed to **one** OlmoEarth forward pass with a `UNetDecoder` head.

Training and prediction run with stock rslearn components, driven by one dataset
config (`data/change_finder_v2/lcc_model_seg/config.json`) and one model config
(`data/change_finder_v2/lcc_model_seg/config.yaml`). The only project-specific
code is a small `BalancedSegmentationHead` (`head.py`) used in place of the stock
`SegmentationHead` (see Loss below).

## Loss

The model uses `BalancedSegmentationHead`, a drop-in replacement for the stock
`SegmentationHead` with identical prediction outputs but a class-balanced loss
(matching the `lcc_model` approach). Within each sample, the cross-entropy is
averaged within each present class, then averaged across classes, then averaged
across samples in the batch. This keeps a dominant class (e.g. 100 `no_change`
points) from swamping a rare transition class (e.g. 1 `deforestation` point).
The `nodata` class (0) is masked out and never contributes.

## Image stack (12 images, one forward pass)

Each window carries a single-instant `time_range` (`T`). The dataset config derives
the imagery from `T` using `time_offset` / `duration` / `period_duration` on the
OlmoEarth Datasets Sentinel-2 source (no Sentinel-1 / Landsat):

- `sentinel2_recent`: `[T, T+2mo]`, 4 least-cloudy mosaics.
- 8-month gap `[T-8mo, T]`: no imagery (this is where the change happens).
- `sentinel2_historical`: `[T-24mo, T-8mo]`, 8 bimonthly least-cloudy mosaics.

The model input lists both layers (`load_all_layers` + `load_all_item_groups`), so
all 12 mosaics are stacked on the time dimension as a single time series. The model
uses `use_legacy_timestamps: true` (dummy sequential month positions), because the
real mosaic timestamps can collide within a modality (two mosaics with the same
center date), which the non-legacy timestamp alignment rejects.

## Classes (num_classes = 11)

- `0` nodata (masked out of the loss; all unlabeled pixels)
- `1` no_change (negative points)
- `2..10` transitions: deforestation, urban_expansion, construction_mining,
  from_water, to_water, urban_erosion, new_crop_field, wetland_loss, forest_regrowth

The transition for each positive point is derived from its
`(pre_category -> post_category)` pair via `TRANSITION_MAP` in `create_windows.py`.
Pairs not in the map are skipped (left nodata).

## Window creation

`create_windows.py` reads v2 annotation JSONs (same format as `lcc_model`) and
creates one 128x128 window + a rasterized `label` layer per entry:

- Positive entries: `T = post_change` of the reference positive point (the first
  usable point: fully annotated, `post_change - pre_change <= 8 months`, and a
  mapped transition). The window is centered on that point so a random 64x64
  training crop reliably contains it.
- Negative-only entries: `T = midpoint(time_range)`, window centered on the first
  negative point.
- Entries with no usable positive point and no negatives are skipped.

The script is idempotent (existing windows are skipped) and assigns a deterministic
`train` / `val` split via a hash of `group/window_name`.

## Workflow

```bash
DS=$RSLP_PREFIX/datasets/change_finder/lcc_model_seg_dataset/
mkdir -p "$DS"
cp data/change_finder_v2/lcc_model_seg/config.json "$DS/config.json"

# 1. Create windows + labels from one or more v2 annotation JSONs.
python -m rslp.change_finder_v2.lcc_model_seg.create_windows \
    --v2-json-paths annotations_a.json annotations_b.json \
    --ds-path "$DS"

# 2. Fetch imagery via the standard pipeline (needs OlmoEarth Datasets creds:
#    OEDATASETS_API_URL, DATASETS_API_TOKEN).
rslearn dataset prepare     --root "$DS" --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root "$DS" --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5 --no-use-initial-job

# 3. Train.
rslearn model fit --config data/change_finder_v2/lcc_model_seg/config.yaml
```

## Prediction

The same config handles prediction. Create windows over an AOI with a single
reference timestamp `T` (in the `predict` group), then prepare/materialize and run
`predict`:

```bash
PREDICT_DS=/path/to/predict_dataset/
mkdir -p "$PREDICT_DS"
cp data/change_finder_v2/lcc_model_seg/config.json "$PREDICT_DS/config.json"

# T is the reference timestamp: the recent stack covers [T, T+2mo], so set T to the
# start of the 2-month window you want as the "after" state.
rslearn dataset add_windows \
    --root "$PREDICT_DS" --group predict --fname aoi.geojson \
    --src_crs EPSG:4326 --utm --resolution 10 --grid_size 2048 \
    --start 2025-04-01T00:00:00+00:00 --end 2025-04-01T00:00:00+00:00

rslearn dataset prepare     --root "$PREDICT_DS" --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root "$PREDICT_DS" --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5 --no-use-initial-job

rslearn model predict \
    --config data/change_finder_v2/lcc_model_seg/config.yaml \
    --data.init_args.path="$PREDICT_DS"
```

The prediction writes a single-band argmax class raster (the predicted transition
class ID per pixel) to the `output` layer of each window.
