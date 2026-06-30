# More OlmoEarth evals: pre/post change classification

This folder adds three OlmoEarth-only evaluation tasks to `rslp.olmoearth_evals`,
all using a new pre/post change-classification task type (`change_classify`):

1. **`forest_loss_driver_prepost`** — the "old-style" forest loss driver: run the
   OlmoEarth encoder separately on the pre-change and post-change image stacks and
   classify the driver from the combined embedding. Uses the existing forest loss
   dataset; no new dataset is created.
2. **`lcc_deforestation`** — tree → X change, derived from `change_finder_v2` v2
   annotations.
3. **`lcc_urban_expansion`** — X → urban/built-up change, derived from the same
   annotations.

## How the `change_classify` task type works

Implemented in [`rslp/olmoearth_evals/olmoearth.py`](../../rslp/olmoearth_evals/olmoearth.py)
(`get_model`, `task_type == "change_classify"`):

- The input is an image time series whose **first half is the pre-change time range**
  and **second half is the post-change time range**. A `SimpleTimeSeries` wraps the
  OlmoEarth encoder with `num_timesteps_per_forward_pass = task_timesteps // 2` and
  `groups=[[0], [1]]`, so each time range is encoded in a single forward pass to
  produce one embedding.
- The two embeddings are combined per the `combine` model-config option (passed via
  the `EVAL_ADAPTER_MODEL_CONFIG` env var, analogous to `decoder`):
  - `"concat"` (default): concatenate → `2 * embedding_size` channels.
  - `"diff"`: signed `post - pre` (via the `ChannelDiff` component) → `embedding_size`
    channels.
- Because each sample is centered on the labeled pixel, the head always uses the
  **center feature only** (`FeatureCenterCrop(sizes=[[1, 1]])`, no spatial pooling),
  followed by a `PoolingDecoder` + `ClassificationHead`.

> **Layer ordering matters:** in the task YAML, list *all* pre-change layers first,
> then *all* post-change layers. The model splits the time series in half.

The tasks are registered in `TASK_CONFIGS` in
[`rslp/olmoearth_evals/launch.py`](../../rslp/olmoearth_evals/launch.py), with task
YAMLs under [`data/olmoearth_evals/tasks/`](../../data/olmoearth_evals/tasks/).

## LCC datasets (deforestation / urban expansion)

Each sample is a 64x64 window centered on a labeled lon/lat point, with a binary
`label` (`"positive"` / `"negative"`) vector layer for the stock `ClassificationTask`.

Positives:
- **deforestation**: a fully-annotated positive point with `pre_category == "tree"`.
- **urban_expansion**: a fully-annotated positive point with
  `post_category == "urban/built-up"`.

All other positive points and all negative points become **negatives**.

To keep the datasets small and avoid negative-heavy entries dominating, **all
positive points are kept but at most 10 negative points per annotation entry** are
used (a deterministic random subset, seeded per entry, when an entry has more).

### Annual embeddings

Both tasks use OlmoEarth annual embeddings for two calendar years:
- **post year** = the year *after* the change date (for positives, the year after
  `post_change`; for negatives, the year after the midpoint of the entry
  `time_range`).
- **pre year** = post year − 3. Points whose pre year would be before **2017** are
  skipped.

The window `time_range` is set to the post calendar year, and
[`config.json`](config.json) derives the imagery from it via the OlmoEarth Datasets
Sentinel-2 source:
- `post_sentinel2`: `time_offset 0d` → 12 monthly mosaics of the post year.
- `pre_sentinel2`: `time_offset -1095d` → 12 monthly mosaics of the year three years
  earlier.

This gives 24 timesteps total (12 pre + 12 post), matching `task_timesteps: 24` in the
LCC task YAMLs.

## Workflow

`create_lcc_windows.py` uses the newer rslearn window-data-storage API, so run it with
the **rslearn repo venv** (the `rslearn_projects` venv may have an older rslearn).

```bash
source /home/favyen/ai2-unison-data/rslearn/.venv/bin/activate

# 1. Create the dataset directory and drop in the shared imagery config.
DS=$RSLP_PREFIX/datasets/olmoearth_evals/lcc_deforestation/
mkdir -p "$DS"
cp one_off_projects/2026_06_26_more_olmoearth_evals/config.json "$DS/config.json"

# 2. Create windows + binary labels from one or more v2 annotation JSONs.
python one_off_projects/2026_06_26_more_olmoearth_evals/create_lcc_windows.py \
    --v2-json-paths annotations_a.json annotations_b.json \
    --ds-path "$DS" \
    --transition deforestation
    # --max-per-class 500   # optional cap per class

# 3. Fetch imagery (needs OlmoEarth Datasets creds: OEDATASETS_API_URL,
#    DATASETS_API_TOKEN).
rslearn dataset prepare     --root "$DS" --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root "$DS" --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5 --no-use-initial-job
```

Repeat steps 1–3 with `--transition urban_expansion` and a
`lcc_urban_expansion/` dataset path. The task YAMLs point at
`/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/{lcc_deforestation,lcc_urban_expansion}/`;
adjust the YAML `path` (or the `$DS` path) to match where you materialize the data.

## Launching evals

The forest loss task needs no new dataset (it reuses the existing forest loss driver
dataset). Launch any of the tasks via the standard `rslp.olmoearth_evals` launcher,
e.g.:

```bash
python -m rslp.main olmoearth_evals launch \
    --models '["olmoearth"]' \
    --tasks '["forest_loss_driver_prepost","lcc_deforestation","lcc_urban_expansion"]' \
    --prefix <prefix> --image_name <image> --project <wandb_project>

# Use the signed difference instead of concatenation:
python -m rslp.main olmoearth_evals launch \
    --models '["olmoearth"]' --tasks '["lcc_deforestation"]' \
    --prefix <prefix> --image_name <image> --project <wandb_project> \
    --model_config '{"combine":"diff"}'
```
