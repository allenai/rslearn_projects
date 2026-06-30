# Embedding Explorer

WARNING: all of the code and most of the documentation here was AI-generated and not
carefully reviewed.

A small Flask app for interactively exploring per-pixel embedding layers on a
Leaflet map. Click points on the map and the server computes a similarity
overlay using one of three modes:

- **Cosine** — cosine similarity to the embedding under the (single, latest)
  positive point.
- **KNN** — fraction of the K nearest labeled points (by cosine similarity)
  that are positive.
- **Linear Probe** — a logistic regression is trained on CPU from the labeled
  points and predict_proba is run over the whole window. Updates only on
  **Apply** (training is fast but not free, so we don't refresh on every click).

Left-click adds a positive point, right-click adds a negative point. The
threshold slider and gradient/threshold toggle are pure client-side rendering
and stay live in all modes.

## Setting up a dataset

The explorer expects an rslearn dataset where each window has at least one
embedding raster layer alongside the source images. Reference configs live in
[`data/embedding_explorer`](../../data/embedding_explorer):

- [config.json](../../data/embedding_explorer/config.json) — minimal dataset
  config (Sentinel-2 L2A from OlmoEarth Datasets).
- [config.yaml](../../data/embedding_explorer/config.yaml) — model config that
  runs OlmoEarth and writes outputs into the
  `embeddings_olmoearth_v1_base_ps4_ws64` layer via `RslearnWriter`.
- [config_olmoearth_10m.yaml](../../data/embedding_explorer/config_olmoearth_10m.yaml)
  — model config that runs OlmoEarth at 10m/pixel into the
  `embeddings_olmoearth_v1_base_ps1_ws16` layer.
- [config_olmoearth_v1_1_base_ps4.yaml](../../data/embedding_explorer/config_olmoearth_v1_1_base_ps4.yaml)
  and [config_olmoearth_v1_1_base_ps1.yaml](../../data/embedding_explorer/config_olmoearth_v1_1_base_ps1.yaml)
  — OlmoEarth v1.1 Base configs that write distinct `embeddings_...` layers.
- [config_olmoearth_v1_1_nano_ps4.yaml](../../data/embedding_explorer/config_olmoearth_v1_1_nano_ps4.yaml)
  and [config_olmoearth_v1_1_nano_ps1.yaml](../../data/embedding_explorer/config_olmoearth_v1_1_nano_ps1.yaml)
  — OlmoEarth v1.1 Nano configs that write distinct `embeddings_...` layers.
- [config_with_aef.json](../../data/embedding_explorer/config_with_aef.json) —
  dataset config that also declares a pre-computed 64-dimensional `aef`
  embedding layer.
- [config_presto.yaml](../../data/embedding_explorer/config_presto.yaml) —
  model config that runs Presto and writes outputs into the `embeddings_presto`
  layer via `RslearnWriter`.
- [config_with_tessera.json](../../data/embedding_explorer/config_with_tessera.json)
  — dataset config that adds Sentinel-1 ascending/descending inputs for Tessera.
- [config_tessera.yaml](../../data/embedding_explorer/config_tessera.yaml) —
  model config that runs Tessera and writes outputs into the
  `embeddings_tessera` layer via `RslearnWriter`.

Model-generated output layer definitions (`embeddings_...`) live in the
corresponding model YAMLs under `RslearnWriter.layer_config`, so the dataset
JSONs only need layers that are prepared or materialized from data sources.

The dataset JSONs use OlmoEarth Datasets-backed sources for Sentinel-2 and
Sentinel-1. Before `rslearn dataset prepare` or `materialize`, make sure
`olmoearth_run[runner]` is importable and set:

```bash
export OEDATASETS_API_URL=https://datasets.olmoearth.allenai.org
export DATASETS_API_TOKEN=<your-token>
```

For a fuller treatment of these configs (Sentinel-1 / Landsat layers, model
sizes, etc.) see
[OlmoEarthEmbeddings.md](https://github.com/allenai/rslearn/blob/master/docs/examples/OlmoEarthEmbeddings.md)
in rslearn.

### 1. Create a dataset and add window(s)

Copy `config.json` into a fresh dataset directory, then add one or more
2048×2048 windows. The explorer is happy with multiple windows and you
switch between them in the sidebar dropdown.

```bash
export DATASET_PATH=./dataset
mkdir -p $DATASET_PATH
cp data/embedding_explorer/config.json $DATASET_PATH/config.json

# A single window over Seattle.
rslearn dataset add_windows --root $DATASET_PATH \
    --group default --name seattle \
    --utm --resolution 10 --src_crs EPSG:4326 \
    --box=-122.5,47.5,-122.2,47.8 \
    --start 2025-01-01T00:00:00+00:00 --end 2026-01-01T00:00:00+00:00

# Or tile a larger area into multiple 2048x2048 windows.
rslearn dataset add_windows --root $DATASET_PATH \
    --group default --name puget_sound \
    --utm --resolution 10 --src_crs EPSG:4326 \
    --box=-122.7,47.2,-122.0,47.9 \
    --start 2025-01-01T00:00:00+00:00 --end 2026-01-01T00:00:00+00:00 \
    --grid_size 2048
```

### 2. Materialize Sentinel-2

```bash
rslearn dataset prepare --root $DATASET_PATH --workers 32 \
    --enabled-layers sentinel2_l2a \
    --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root $DATASET_PATH --workers 32 \
    --no-use-initial-job --enabled-layers sentinel2_l2a \
    --retry-max-attempts 5 --retry-backoff-seconds 5
```

### 2b. Materialize AEF embeddings (optional)

AEF provides pre-computed 64-dimensional satellite embeddings at 10m resolution
from [source.coop/tge-labs/aef](https://source.coop/tge-labs/aef). Use
`config_with_aef.json` instead of `config.json`:

```bash
cp data/embedding_explorer/config_with_aef.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --workers 32 --enabled-layers aef
rslearn dataset materialize --root $DATASET_PATH --workers 32 \
    --no-use-initial-job --enabled-layers aef
```

### 3. Compute OlmoEarth embeddings (40m, default)

```bash
rslearn model predict --config data/embedding_explorer/config.yaml
```

After this finishes, each window will have a populated
`layers/embeddings_olmoearth_v1_base_ps4_ws64/` directory with a 768-band
GeoTIFF at 40m/pixel (patch_size=4).

### 3b. Compute OlmoEarth embeddings at 10m (optional)

For a fair comparison with AEF (both at 10m), use `config_olmoearth_10m.yaml`
which runs OlmoEarth-v1-Base with `patch_size=1` (one embedding per 10m pixel):

```bash
rslearn model predict --config data/embedding_explorer/config_olmoearth_10m.yaml
```

This writes to `embeddings_olmoearth_v1_base_ps1_ws16`, so it can coexist with
the 40m output in the same dataset.

### 3c. Compute OlmoEarth v1.1 embeddings (optional)

The v1.1 Base and Nano configs write to distinct output layers, so you can run
multiple variants in the same dataset:

```bash
rslearn model predict --config data/embedding_explorer/config_olmoearth_v1_1_base_ps4.yaml
rslearn model predict --config data/embedding_explorer/config_olmoearth_v1_1_base_ps1.yaml
rslearn model predict --config data/embedding_explorer/config_olmoearth_v1_1_nano_ps4.yaml
rslearn model predict --config data/embedding_explorer/config_olmoearth_v1_1_nano_ps1.yaml
```

Pass the corresponding output layer names to the app, such as
`embeddings_olmoearth_v1_1_base_ps4_ws64` or
`embeddings_olmoearth_v1_1_nano_ps1_ws16`.

### 3d. Compute Presto embeddings (optional)

Presto is supported in rslearn as `rslearn.models.presto.Presto`. The provided
config uses Sentinel-2 only, so `config.json` is enough. Materialize Sentinel-2 as
above and run:

```bash
rslearn model predict --config data/embedding_explorer/config_presto.yaml
```

This writes 128-dimensional embeddings at 10m/pixel to the `embeddings_presto`
layer. The provided config uses Sentinel-2 only, matching the default explorer
dataset.
Presto can also consume Sentinel-1 when the dataset has a compatible `s1` input.
The first run may need to download or otherwise populate the Presto checkpoint
cache used by `rslearn.models.presto.Presto`.

### 3e. Compute Tessera embeddings (optional)

Tessera is supported in rslearn as `rslearn.models.tessera.Tessera`. Use
`config_with_tessera.json` instead of `config.json` when creating the dataset.
Tessera uses Sentinel-2 plus separate ascending and descending Sentinel-1 RTC
time series, so materialize all three source layers:

```bash
cp data/embedding_explorer/config_with_tessera.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --workers 32 \
    --enabled-layers sentinel2_l2a,sentinel1_ascending,sentinel1_descending \
    --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root $DATASET_PATH --workers 32 \
    --no-use-initial-job \
    --enabled-layers sentinel2_l2a,sentinel1_ascending,sentinel1_descending \
    --retry-max-attempts 5 --retry-backoff-seconds 5
```

Download an encoder-only Tessera v1.1 checkpoint and point the model config at
it, then run prediction:

```bash
export TESSERA_CHECKPOINT_PATH=/path/to/tessera_v1_1_mpc_encoder.pt
rslearn model predict --config data/embedding_explorer/config_tessera.yaml
```

This writes 192-dimensional Tessera embeddings at 10m/pixel to the
`embeddings_tessera` layer. The provided config converts the OlmoEarth Datasets
Sentinel-1 RTC layers to standard dB with `Sentinel1ToDecibels`, then applies
`TesseraNormalize(data_source="mpc")` before the model runs. If your Sentinel-1
layers are already stored in standard dB, skip `Sentinel1ToDecibels` and still
run `TesseraNormalize` so the model receives checkpoint-ready normalized inputs.

## Run the app

The app needs Flask and scikit-learn (neither are pulled in by `rslp`'s base
requirements):

```bash
pip install flask scikit-learn
```

Then point it at your dataset:

```bash
python -m rslp.embedding_explorer.app \
    --dataset-path $DATASET_PATH \
    --embedding-layer embeddings_olmoearth_v1_base_ps4_ws64 \
    --port 5000
```

To load multiple embedding layers (e.g. OlmoEarth + AEF), pass them all:

```bash
python -m rslp.embedding_explorer.app \
    --dataset-path $DATASET_PATH \
    --embedding-layer embeddings_olmoearth_v1_base_ps4_ws64 aef \
    --port 5000
```

When multiple layers are loaded, a dropdown appears in the sidebar to select
which embedding is used for similarity queries. Each layer can have a different
resolution — the overlay adapts to the selected layer's grid.

For example, to compare OlmoEarth and Presto:

```bash
python -m rslp.embedding_explorer.app \
    --dataset-path $DATASET_PATH \
    --embedding-layer embeddings_olmoearth_v1_base_ps4_ws64 embeddings_presto \
    --port 5000
```

Or compare OlmoEarth and Tessera:

```bash
python -m rslp.embedding_explorer.app \
    --dataset-path $DATASET_PATH \
    --embedding-layer embeddings_olmoearth_v1_base_ps4_ws64 embeddings_tessera \
    --port 5000
```

Open `http://localhost:5000` and pick a window from the sidebar. Clicking on
the map adds points; in cosine/KNN modes the overlay updates immediately, and
in Linear Probe mode you press **Apply** to (re)train.
