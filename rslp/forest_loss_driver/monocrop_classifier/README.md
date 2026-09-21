# Monocrop classifier

This is a separate time-series classification model from the forest-loss driver
classifier. Given a forest loss event polygon and monthly Sentinel-2 imagery after
the loss event, it predicts the crop/land-use class of the event.

The dataset config and model config are in
`data/forest_loss_driver/monocrop_classifier/`. The README there records the
experiments (label hierarchy, frozen vs. LLRD fine-tuning, segmentation vs.
classification, and pooling strategies) that led to `model_classify_pool.yaml`.

## Annotation filtering

The default source projects (`DEFAULT_PROJECT_IDS` in `studio.py`) are Monocrop -
Peru, Monocrop - Bolivia, and Monocrop - Ecuador; pass `--project-id` to
`create_dataset.py` to use other projects, such as the phase-2 project populated
by `scripts/phase2_upload_to_studio.py`. Use annotations that have:

- a `monoculture_tag` in the confirmed class list;
- `confidence` equal to `high`, `medium`, or `low`;
- annotation status other than `rejected`;
- a valid polygon and task event timestamp.

Class IDs and filtering values are constants in `create_dataset.py`. The Studio
labelset has eight raw classes; the label written to the dataset merges
`soybean` into the `mennonites_soybean` slot (renamed `soybean`), giving the
seven classes the model predicts:

| ID | Raw Studio class | Dataset class |
|---:|---|---|
| 0 | nodata | nodata |
| 1 | mennonites_nonsoybean | mennonites_nonsoybean |
| 2 | mennonites_soybean | soybean |
| 3 | oil_palm | oil_palm |
| 4 | other_agriculture | other_agriculture |
| 5 | pastures | pastures |
| 6 | rice | rice |
| 7 | soybean | soybean (merged into 2) |

`nodata` is never a label; it is kept so class indices stay stable. Each window
gets one feature in the `label_vector` layer whose geometry is the annotation
polygon and whose `class_name`/`class_id` are the merged class, with
`raw_class_name`/`raw_class_id` recording the Studio label.

## Create and materialize the dataset

The source stack has 11 least-cloudy 30-day Sentinel-2 mosaics before the event
and 1-12 complete periods after it. The default imagery cutoff is in
`create_dataset.py` (`DEFAULT_IMAGERY_CUTOFF`). The cutoff determines the maximum
usable post-loss month for each record, so records become windows even when
fewer than 12 months have elapsed since the event.

From the `rslearn_projects` repository:

```bash
DS_PATH=/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/monocrop_classifier/20260914/
python -m rslp.forest_loss_driver.monocrop_classifier.create_dataset \
  --ds-path "$DS_PATH" \
  --imagery-cutoff 2026-08-27T00:00:00Z
rslearn dataset prepare --root "$DS_PATH" --workers 32
rslearn dataset materialize --root "$DS_PATH" --workers 32
```

After materialization, each selected window must have `11 + max_post_months` item
groups in `sentinel2_l2a`. Each window stores `max_post_months` in its metadata,
and dataset creation prints aggregate horizon counts. A missing interior month
should be investigated instead of padded.

Windows are in per-country groups (`peru`, `bolivia`, `ecuador`, ...) derived from
the Studio project name, and the train/val split is assigned deterministically
from the source event geometry and time.

## Train

The training transform (`PostLossMonthSampler`) samples an elapsed month
uniformly from 1 through the maximum available for that window and always sends
12 frames to OlmoEarth-v1.2-Base. An elapsed month `m` gives the model `12-m`
pre-loss frames followed by the first `m` post-loss frames.

```bash
rslearn model fit \
  --config data/forest_loss_driver/monocrop_classifier/model_classify_pool.yaml
```

The model max-pools the OlmoEarth feature map over the whole window
(`PoolingDecoder`) followed by a linear classification head, and fine-tunes the
encoder with layer-wise learning-rate decay. Validation uses the six-month view;
if a window has fewer than six post-loss months, the transform uses its maximum
available month instead. `test_config` intentionally reuses the validation split.

`val_accuracy` is micro-averaged, so it is the fraction of windows predicted
correctly, and a confusion matrix is logged to wandb. Checkpointing monitors
`val_accuracy`.

## Test elapsed months

Set `MONOCROP_NUM_POST_MONTHS` from 1 through 12. It defaults to 6 when unset. A
request above a window's available horizon is clamped for that window; for example,
an eight-month test uses six months on a window whose maximum is six.

```bash
MONOCROP_NUM_POST_MONTHS=1 rslearn model test \
  --config data/forest_loss_driver/monocrop_classifier/model_classify_pool.yaml \
  --ckpt_path /path/to/checkpoint.ckpt

MONOCROP_NUM_POST_MONTHS=12 rslearn model test \
  --config data/forest_loss_driver/monocrop_classifier/model_classify_pool.yaml \
  --ckpt_path /path/to/checkpoint.ckpt
```

Use the same checkpoint and validation dataset for all month values.

## Create a prediction dataset

`create_prediction_dataset.py` builds a separate prediction dataset from a GeoJSON
of forest loss event polygons (e.g. the output of
`../scripts/monocrop_initial_setup_20260624/make_annotation_sets.py`). Each feature
must have polygon geometry and an `oe_start_time` property (the event time); the
other properties (`tif_fname`, `center_pixel`, `oe_end_time`, `country`,
`new_label`, `probs`, `area_ha`) are copied into `window.options` when present.

```bash
DS_PATH=/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/monocrop_classifier/predict_20260721/
python -m rslp.forest_loss_driver.monocrop_classifier.create_prediction_dataset \
  --geojson /path/to/polygons.geojson \
  --ds-path "$DS_PATH"
rslearn dataset prepare --root "$DS_PATH" --workers 32
rslearn dataset materialize --root "$DS_PATH" --workers 32
```

Each window goes in group `predict`, is a 128x128 pixel 10 m UTM window centered
on the polygon (same as training windows), and spans `oe_start_time` through
`oe_start_time` plus 360 days (12 30-day post-loss periods, no pre-loss
coverage). Prediction omits `PostLossMonthSampler`, so the un-sampled 12-frame
stack is exactly the 12-month elapsed view from training, which uses zero
pre-loss frames. No label layer is written. Window names are deterministic
digests of `(tif_fname, center_pixel, oe_start_time)`, so re-running the script
skips existing windows and duplicate features collapse to one window.

There is no recency filter in the script: `min_matches` is 12 in the dataset
config, so a window whose event is too recent to have all 12 complete post-loss
months matches fewer than 12 periods and is not materialized, and `model predict`
skips it because its `sentinel2_l2a` layer is incomplete.

## Predict

Prediction deliberately omits `PostLossMonthSampler`. Windows must be in group
`predict` and provide exactly the 12 post-loss monthly item groups under
`sentinel2_l2a` (see the section above):

```bash
rslearn model predict \
  --config data/forest_loss_driver/monocrop_classifier/model_classify_pool.yaml \
  --data.init_args.path "$DS_PATH"
```

The `RslearnWriter` callback writes one feature per window to the `output_vector`
layer with the predicted `class_name` and per-class `probs`.

## Phase-2 annotation sampling

`scripts/phase2_upload_to_studio.py` samples prediction windows per predicted
class and uploads them as tasks to a new Studio project for the next annotation
round. See its module docstring for details.
