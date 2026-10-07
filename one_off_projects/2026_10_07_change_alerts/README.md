## Change Alert Experiments

Experiments on near-real-time change alerts with OlmoEarth v1.2 Base: the model should
detect a change within a few weeks of when it becomes observable, and predict both the
change category and the timestep at which it appears. The training data comes from
the "Ai2 - Change Detection Experimentation" organization on OlmoEarth Studio, where
each annotation has a location, change date, and change category. We train
separately on three of its projects (Forest Loss Driver, LCC, and Mangrove; Nepal is
excluded since its change dates are only known to the year).

Annotations are only used if:

- Their change date is between 2019-01-01 (so that there are 900 days of Sentinel-2 L2A
  history before it) and 2026-07-02 (97 days before the datasets were created, so that
  the whole detection range has imagery).
- Their category has at least 50 annotations within that date range. This drops
  airstrip from Forest Loss Driver and landslide, new_infrastructure, and
  selective_logging from LCC.
- For LCC, their task is not from one of the batches with about five negatives per
  window (see `skip_source_files` in `common.py`). The LCC Studio project was created
  from the output of `rslp/olmoearth_lcc/lcc_model/filter_v2_jsons.py` (on the
  `favyen/20260407-change-finder` branch), so it already
  excludes entries that fail the olmoearth_lcc date checks or have no points.

The generic components are in rslearn under `rslearn.change_alerts` (see
`rslearn/docs/ChangeAlerts.md`):

- `slots.py`: the dataset layers for time series ("slots") that end at given offsets
  after the change date.
- `sampler.py`: `ChangeTimeSeriesSampler`, which builds one time series per example
  from a slot and derives the category and timestep targets.
- `metrics.py`: balanced accuracy, change-vs-none AUROC, and timestep accuracy within
  a tolerance.

The Beaker image used for training must include this version of rslearn.

### Experiments

The change detection range is 90 days. Each training example ends at an offset after
the change between 7 and 90 days.

| Run | Slot augmentation | Context | Category head |
| --- | --- | --- | --- |
| `fixed4_history_pool` (baseline) | 4 fixed slots | history | attention pool |
| `rand4_history_pool` | 4 randomized slots | history | attention pool |
| `rand1_history_pool` | 1 randomized slot | history | attention pool |
| `fixed4_recent_pool` | 4 fixed slots | recent | attention pool |
| `fixed4_history_breakpoint` | 4 fixed slots | history | BreakpointScan |
| `fixed4_recent_breakpoint` | 4 fixed slots | recent | BreakpointScan |

- Fixed slots: the time series ends 7, 35, 62, or 90 days after the change (evenly
  spaced from one week to the detection range), and one slot is picked at random
  for each training example.
- Randomized slots: each window has its own end offsets drawn uniformly from 7 to 90
  days, as in the LCC model. rand4 has four per window, rand1 only one.
- History context: 8 quarterly (90-day) mosaics followed by the latest 4 weekly
  mosaics within the last 60 days. The quarterly mosaics end 60 days before the end of
  the time series, so if the 4 weekly mosaics are the latest 4 weeks, there is a gap
  of about a month between the quarterly and weekly mosaics.
- Recent context: the latest 20 weekly mosaics (within the last 26 weeks).
- Category head: attention pooling over the timesteps (`SimpleAttentionPool`), or the
  before/after features from a `BreakpointScan`.

All runs predict the timestep of the change with `TokensToChannels` and
`PerPixelTimestepHead` (12 timesteps for the history context, 20 for the recent
context). The arms are compared on the change category; the timestep metrics are
logged too.

All runs validate on randomized slot 0, so validation covers the whole detection range
and is the same across runs. The best checkpoint is selected by
`val_category/balanced_accuracy`.

### Test Scenarios

Each run is tested on time series ending E = 7, 45, and 90 days after the change: the
change is only in the latest image, about 6 weeks old, and at the end of the detection
range. There are two test datasets per source, containing the same test split windows:

- test_history: weekly mosaics covering 63 days and quarterly mosaics for each E.
- test_recent: weekly mosaics covering 26 weeks for each E.

### Paths

Everything is on WEKA under `/weka/dfive-default/rslearn-eai/datasets/change_alerts/20261007/`:

- `{source}/train`: the training dataset (train and val splits).
- `{source}/test_history`: the test dataset for history context runs.
- `{source}/test_recent`: the test dataset for recent context runs.

`{source}` is one of `forest_loss`, `lcc`, and `mangrove`.

Checkpoints are under `${RSLP_PREFIX}/projects/2026_10_07_change_alerts/{source}_{run}/`.

### Building the Datasets

These environment variables are needed:

- `STUDIO_API_KEY` (see `rslearn_projects/.env`) for creating the windows.
- `OEDATASETS_API_URL` and `DATASETS_API_TOKEN` for the OlmoEarth Datasets Sentinel-2
  L2A data source used by prepare and materialize (see `rslearn/.env`). Materialization
  jobs launched with `rslp.main common launch_data_materialization_jobs` need them
  passed explicitly with `--extra_env_vars` and `--extra_env_secrets`.

Windows are created and prepared locally, then materialized on Beaker. For each source,
from this directory (a JSON cache of the Studio tasks and annotations avoids fetching
them three times; the filters above are applied after loading it):

```
SOURCE=forest_loss
ROOT=/weka/dfive-default/rslearn-eai/datasets/change_alerts/20261007/$SOURCE
CACHE=/weka/dfive-default/rslearn-eai/datasets/change_alerts/20261007/annotations_$SOURCE.json

# Training dataset.
python make_config.py --mode train --ds_path $ROOT/train
python create_windows.py --source $SOURCE --kind train --annotations_cache $CACHE --workers 32
rslearn dataset prepare --root $ROOT/train --workers 64 --retry-max-attempts 3 \
    --enabled-layers freq_fixed_0,infreq_fixed_0,freq_fixed_1,infreq_fixed_1,freq_fixed_2,infreq_fixed_2,freq_fixed_3,infreq_fixed_3
python prepare_randomized.py --ds_path $ROOT/train --workers 64

# Test datasets.
for KIND in test_history test_recent; do
    python make_config.py --mode $KIND --ds_path $ROOT/$KIND
    python create_windows.py --source $SOURCE --kind $KIND --annotations_cache $CACHE --workers 32
    rslearn dataset prepare --root $ROOT/$KIND --workers 64 --retry-max-attempts 3
done
```

The randomized layers (`freq_rand_{k}` and `infreq_rand_{k}`) must only be prepared
with `prepare_randomized.py`, not `rslearn dataset prepare`: in config.json they end at
the change date, and `prepare_randomized.py` shifts each window by its random offset
when preparing them.

Materialization is slow: each item group (one mosaic) takes about 8 seconds per worker,
since each of the 12 bands is a separate COG read (plus an OlmoEarth Datasets API
lookup). A training window has about 280 item groups, and each window is materialized
by one worker, so it takes over 30 minutes. The windows are spread over 16 groups per
source (`{source}_00` to `{source}_15`), and each Beaker job materializes one group
index across all nine dataset roots, e.g. for group 03:

```
for SOURCE in forest_loss lcc mangrove; do
    for KIND in train test_history test_recent; do
        rslearn dataset materialize --root $DATASET_ROOT/$SOURCE/$KIND --group ${SOURCE}_03 \
            --workers 64 --no-use-initial-job --retry-max-attempts 2 --retry-backoff-seconds 10 \
            --ignore-errors
    done
done
```

Use `--no-use-initial-job`, otherwise the first window is materialized serially before
the workers start.

The OlmoEarth Datasets Sentinel-2 items were being renamed in October 2026 (appending
`_S<time>` to the name). Items that are renamed between prepare and materialize fail
with "Expected 1 item for X, got 0 from OlmoEarth API", leaving that layer incomplete.
`fix_incomplete.py` finds incomplete layers and prepares them again; then run
materialize again, and repeat until no layers are incomplete:

```
python fix_incomplete.py --ds_path $ROOT/train
rslearn dataset materialize --root $ROOT/train --workers 64 --no-use-initial-job \
    --retry-max-attempts 2 --retry-backoff-seconds 10
```

### Training

`make_model_configs.py` writes the training configs to `configs/{source}/{run}.yaml`
(they are checked in, so this is only needed after changing it). Then launch all 18
runs (or a subset with `--sources` and `--runs`) on Beaker:

```
python make_model_configs.py
python launch.py --image_name [BEAKER_IMAGE] --dry_run
python launch.py --image_name [BEAKER_IMAGE]
```

### Evaluation

`eval_test.py` tests the best checkpoint of each run on each scenario and writes the
metrics to a CSV. It runs locally (with a GPU and RSLP_PREFIX set), e.g. in a Beaker
session with WEKA mounted:

```
python eval_test.py --out results.csv
```

The metrics are:

- `test_category/balanced_accuracy` (headline): mean per-category recall.
- `test_category/accuracy`: pixel accuracy of the category.
- `test_category/change_auroc`: AUROC of change versus no change.
- `test_timestep/accuracy` and `test_timestep/within1`: accuracy of the predicted
  change timestep at change pixels, exactly or within one timestep.

### Results

Category balanced accuracy on the test set, for E = 7 / 45 / 90 days.

| Run | Forest Loss Driver | LCC | Mangrove |
| --- | --- | --- | --- |
| `fixed4_history_pool` | | | |
| `rand4_history_pool` | | | |
| `rand1_history_pool` | | | |
| `fixed4_recent_pool` | | | |
| `fixed4_history_breakpoint` | | | |
| `fixed4_recent_breakpoint` | | | |
