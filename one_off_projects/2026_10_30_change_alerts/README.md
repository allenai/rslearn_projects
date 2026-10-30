## Change Alert Experiments, Round 3 (Unified Config)

Round 3 of the change alert experiments (rounds 1 and 2 are in
`one_off_projects/2026_10_07_change_alerts`). rslearn now uses one dataset config for
training and prediction (`rslearn.change_alerts.layers`, see
`rslearn/docs/ChangeAlerts.md`), so this round checks that the new sampler performs at
least as well as the round 2 baseline (`fixed4_history_pool`, four fixed slots).

The annotations, filters, categories, train/val/test splits, and window groups are the
same as in rounds 1 and 2 (Forest Loss Driver, LCC, and Mangrove projects of the "Ai2 -
Change Detection Experimentation" Studio organization), so each annotation has the same
window location.

### Dataset

One dataset per source, `{DATASET_ROOT}/{source}`, with one `config.json`
(`make_config.py`) for all windows:

- `frequent`: weekly Sentinel-2 L2A mosaics covering the window time range.
- `infrequent`: the 8 quarterly mosaics before the window start.
- `label_category`, `label_change_day`: the labels, as in rounds 1 and 2.
- `output_category`, `output_change_day`: prediction outputs.

With `ChangeAlertTiming(7 days, 4 frequent images, 91-day detection range)`:

- Train split: one training window per annotation, from 21 days before to 91 days after
  the change (16 weekly mosaics).
- Val and test splits: one prediction window per annotation and scenario E (7, 45, and
  90 days after the change), covering the 4 weeks before E days after the change. They
  are named `{annotation ID}_e{E}` and have the `end_days` option set to E.

### Runs

`history_pool` with seeds 0, 1, and 2 (rounds 1 and 2 used seed 0), for each source.
The model is the same as the round 2 `fixed4_history_pool` baseline (OlmoEarth v1.2
Base, attention-pooled category head, per-timestep timestep head, 8 quarterly + 4 weekly
mosaics). `ChangeTimeSeriesSampler` picks 4 consecutive weekly mosaics at a random
position in each training window (so the change is 1 to 13 weeks before the end of the
time series), and keeps only the latest 1-3 of them half of the time. Validation uses
the val prediction windows with `select: latest`, and the best checkpoint is selected by
`val_category/balanced_accuracy`.

### Test Sets

`eval_test.py` tests round 3 runs and round 1/2 history runs (whose checkpoints are
tested with the round 3 sampler) on two test sets:

- `round3`: the test split prediction windows, per scenario E.
- `round2`: the round 2 `test_history` datasets, where the weekly mosaics are the latest
  4 within 63 days and the quarterly mosaics are anchored 27 days after the change.
  Round 2 runs reproduce their round 2 results on it, and round 3 runs can be compared
  with them on the same inputs.

The metrics are the same as in rounds 1 and 2 (headline:
`test_category/balanced_accuracy`).

### Commands

`STUDIO_API_KEY`, `OEDATASETS_API_URL`, and `DATASETS_API_TOKEN` are needed as in rounds
1 and 2. From this directory:

```
SOURCE=forest_loss
ROOT=/weka/dfive-default/rslearn-eai/datasets/change_alerts/20261030/$SOURCE
CACHE=/weka/dfive-default/rslearn-eai/datasets/change_alerts/20261012/annotations_$SOURCE.json
python make_config.py --ds_path $ROOT
python create_windows.py --source $SOURCE --annotations_cache $CACHE --workers 32
rslearn dataset prepare --root $ROOT --workers 64 --retry-max-attempts 3
rslearn dataset materialize --root $ROOT --group ${SOURCE}_03 --workers 64 \
    --no-use-initial-job --retry-max-attempts 3 --ignore-errors
```

Training and evaluation (the Beaker image must include rslearn with
`rslearn.change_alerts.layers`):

```
python make_model_configs.py
python launch.py --image_name [BEAKER_IMAGE] --priority urgent
python launch_eval.py --image_name [BEAKER_IMAGE] --priority urgent --out_dir [OUT_DIR]
```
