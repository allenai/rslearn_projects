# Embedding Explorer

WARNING: all of the code and most of the documentation here was AI-generated and not
carefully reviewed.

A small Flask app for interactively exploring per-pixel OlmoEarth embeddings
on a Leaflet map. Click points on the map and the server computes a similarity
overlay using one of three modes:

- **Cosine** — cosine similarity to the embedding under the (single, latest)
  positive point.
- **KNN** — fraction of the K nearest labeled points (by cosine similarity)
  that are positive.
- **Linear Probe** — a logistic regression (768 -> 2) is trained on CPU from the
  labeled points and predict_proba is run over the whole window. Updates only
  on **Apply** (training is fast but not free, so we don't refresh on every
  click).

Left-click adds a positive point, right-click adds a negative point. The
threshold slider and gradient/threshold toggle are pure client-side rendering
and stay live in all modes.

## Setting up a dataset

The explorer expects an rslearn dataset where each window has a raster layer
named `embeddings` (768 bands, float32) alongside the source images. Two
reference configs sit next to this README:

- [config.json](config.json) — minimal dataset config (Sentinel-2 L2A from
  Planetary Computer + an `embeddings` layer).
- [config.yaml](config.yaml) — model config that runs OlmoEarth and writes
  outputs into the `embeddings` layer via `RslearnWriter`.

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
cp rslp/embedding_explorer/config.json $DATASET_PATH/config.json

# A single 2048x2048 window over Seattle.
rslearn dataset add_windows --root $DATASET_PATH \
    --group default --name seattle \
    --utm --resolution 10 --src_crs EPSG:4326 \
    --box=-122.42,47.58,-122.22,47.78 \
    --start 2024-06-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 \
    --grid_size 2048

# Or tile a larger area into multiple 2048x2048 windows.
rslearn dataset add_windows --root $DATASET_PATH \
    --group default --name puget_sound \
    --utm --resolution 10 --src_crs EPSG:4326 \
    --box=-122.7,47.2,-122.0,47.9 \
    --start 2024-06-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00 \
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

### 3. Compute embeddings

```bash
rslearn model predict --config rslp/embedding_explorer/config.yaml
```

After this finishes, each window will have a populated `layers/embeddings/`
directory with a 768-band GeoTIFF.

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
    --port 5000
```

Open `http://localhost:5000` and pick a window from the sidebar. Clicking on
the map adds points; in cosine/KNN modes the overlay updates immediately, and
in Linear Probe mode you press **Apply** to (re)train.
