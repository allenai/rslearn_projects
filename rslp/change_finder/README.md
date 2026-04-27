# Change Finder

Self-supervised change detection from multi-year Sentinel-2 imagery. The model
learns to distinguish "more change" from "less change" between image pairs
without any human labels.

## Idea

Given a location and a 7-year span of Sentinel-2 imagery (years 0 through 6),
we construct two training triplets:

| Triplet | anchor1 | query | anchor2 | Label |
|---------|---------|-------|---------|-------|
| A       | y1      | y0    | y6      | 0     |
| B       | y0      | y6    | y5      | 1     |

**Class 0** means the query is temporally close to anchor1 (1-year gap) and far
from anchor2 (6-year gap). **Class 1** is the reverse: close to anchor2 and far
from anchor1. A well-trained model should output higher P(class 1) when the
query image shows more change relative to the anchors.

At evaluation time, we fix anchor1=y0 and anchor2=y6, then sweep query through
y1..y5. A location with a real change event between years N and N+1 should show
a spike in P(class 1) at that point.

## Architecture

`ChangeFinderModel` is a dual-encoder binary classifier:

1. Concatenate (anchor1, query) along the time axis → **pair_a**
2. Concatenate (anchor2, query) along the time axis → **pair_b**
3. Run each pair through a shared **OlmoEarth-v1-Base** encoder
4. Global max-pool each encoder output → two feature vectors (768-d each)
5. Concatenate and classify with 2 FC layers (768*2 → 512 → 2)

The encoder is frozen for the first 10 epochs, then unfrozen with a 10x lower
learning rate.

## Dataset

Each Sentinel-2 layer retrieves 4 monthly composites (one per 30-day period)
within a 120-day window, sorted by cloud cover. The 7 layers are offset by
0, 1, 2, ..., 6 years from the base window time.

- **Data source:** Planetary Computer Sentinel-2 L2A (harmonized)
- **Bands:** B01-B12 + B8A (no B10) — 12 bands, uint16
- **Spatial:** 128×128 px at 10 m/pixel in UTM
- **Temporal:** base window randomly chosen from Jan 2016 – Dec 2019; the
  6-year offset extends to at most ~Apr 2026

The band order used by OlmoEarth normalization is:
`B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09`

## Files

| File | Purpose |
|------|---------|
| `data/change_finder/config.json` | rslearn dataset config — defines the 7 `sentinel2_yN` layers |
| `data/change_finder/config.yaml` | Training config (model, data module, trainer) |
| `rslp/change_finder/__init__.py` | Package init, registers `create_windows` workflow |
| `rslp/change_finder/create_windows.py` | Samples random land points, verifies S2 coverage, saves windows |
| `rslp/change_finder/train.py` | `ChangeFinderTransform`, `ChangeFinderModel`, `ChangeFinderTask`, `ChangeFinderLightningModule` |
| `rslp/change_finder/evaluate.py` | Loads checkpoint, runs y1..y5 triplets, writes per-window scores to JSON |

## Usage

### 1. Create dataset windows

```bash
python -c "
from rslp.change_finder import workflows
workflows['create_windows'](
    ds_path='gs://rslearn-eai/datasets/change_finder/dataset_v1/20260403/',
    num_samples=50000,
    workers=32,
)
"
```

This samples 50k random (lat, lon) points, filters to land with S2 coverage
(at least one image per 30-day period in the base window), snaps to a 128px
UTM grid, and saves the windows. Roughly 30% of samples survive the ocean
filter; coverage filtering removes additional points.

### 2. Prepare (materialize imagery)

Use the standard rslearn pipeline to fetch Sentinel-2 images for each window:

```bash
rslearn dataset prepare --ds-path gs://rslearn-eai/datasets/change_finder/dataset_v1/20260403/
```

### 3. Train

```bash
python -m rslp.change_finder.train fit \
    --config data/change_finder/config.yaml \
    --data.init_args.path gs://rslearn-eai/datasets/change_finder/dataset_v1/20260403/
```

Only y0, y1, y5, y6 are loaded during training. The `ChangeFinderTransform`
randomly picks one of the two triplets per sample and assigns the binary label.

### 4. Evaluate

```bash
python -m rslp.change_finder.evaluate \
    --ds_path gs://rslearn-eai/datasets/change_finder/dataset_v1/20260403/ \
    --checkpoint_path /path/to/best.ckpt \
    --output_path results.json
```

All 7 year layers are loaded during evaluation. For each val window, the script
computes P(class 1) for triplets `[y0, yX, y6]` with X in {1..5} and writes:

```json
[
  {
    "window_name": "12.3456_78.9012",
    "window_group": "default",
    "base_time": "2017-06-01T00:00:00+00:00",
    "scores": {"y1": 0.12, "y2": 0.25, "y3": 0.41, "y4": 0.78, "y5": 0.91}
  }
]
```

A spike from low to high values at some year X suggests a change event occurred
around that time. How to aggregate these into a single change score is TBD.

## Key design decisions

- **Dual encoder, not concatenated triplet:** Processing (anchor, query) pairs
  separately limits the model's ability to "cheat" by comparing all three inputs
  at once. It must extract meaningful change features from each pair independently.
- **Self-supervised labels:** No human annotation. The assumption is that 1-year
  temporal neighbors are more similar than 6-year neighbors, which holds on
  average even though individual locations may violate this.
- **4 composites per period:** Redundancy against cloud cover. OlmoEarth handles
  multi-temporal inputs natively via its temporal encoding.
- **Coverage check at window creation:** Each candidate point is verified against
  the Planetary Computer STAC API to ensure at least one S2 image exists in
  every 30-day period of the base window. This avoids creating windows that will
  fail during materialization.
