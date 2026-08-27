# Monocrop classifier

This is a separate time-series segmentation model from the forest-loss driver
classifier. It predicts the crop/land-use class inside an annotated forest-loss
polygon using monthly Sentinel-2 imagery after the loss event.

## Annotation filtering

The source projects are Monocrop - Peru, Monocrop - Bolivia, and Monocrop -
Ecuador. Use annotations that have:

- a `monoculture_tag` in the confirmed class list;
- `confidence` equal to `high`, `medium`, or `low`;
- annotation status other than `rejected`;
- a valid polygon and task event timestamp.

The Studio inventory contained 542 annotations. The filtering accepts 536 metadata
records: 535 approved and one pending. It excludes five rejected/unlabeled records
and one approved record with missing confidence.

Class IDs and filtering values are constants in `create_dataset.py`:

| ID | Class |
|---:|---|
| 0 | nodata |
| 1 | mennonites_nonsoybean |
| 2 | mennonites_soybean |
| 3 | oil_palm |
| 4 | other_agriculture |
| 5 | pastures |
| 6 | rice |
| 7 | soybean |

Class 0 is outside the annotation polygon and is masked from loss and metrics.
The annotation polygon, rather than the larger enclosing Studio task polygon, is
rasterized as the target.

## Create and materialize the dataset

The source stack has 11 least-cloudy 30-day Sentinel-2 mosaics before the event
and 1-12 complete periods after it. The default imagery cutoff is 2026-07-20. The
cutoff determines the maximum usable post-loss month for each record, so all 536
accepted records become windows even when fewer than 12 months have elapsed.

From the `rslearn_projects` repository:

```bash
DS_PATH=/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/monocrop_classifier_20260720/
python -m rslp.forest_loss_driver.monocrop_classifier.create_dataset \
  --ds-path "$DS_PATH" \
  --imagery-cutoff 2026-07-20T00:00:00Z
rslearn dataset prepare --root "$DS_PATH" --workers 32
rslearn dataset materialize --root "$DS_PATH" --workers 32
```

After materialization, each selected window must have `11 + max_post_months` item
groups in `sentinel2_l2a`. Each window stores `max_post_months` in its metadata,
and dataset creation prints aggregate horizon counts. A missing interior month
should be investigated instead of padded.

## Tag windows

rslearn tag filtering only supports exact key/value matches on `window.options`,
so derived window subsets must be precomputed as options. `tag_windows.py` adds
`class_group: soy` to windows whose class is `mennonites_soybean` or `soybean`
and `class_group: other` to the rest; the soy-only experiment config references
it with `tags: {class_group: soy}`. Boolean-looking tag values like `"true"`
must be avoided because jsonargparse coerces them to Python booleans, which
never match the string stored in window options. Country filtering needs no
tagging because windows are already in per-country groups (`peru`, `bolivia`,
`ecuador`).

```bash
python -m rslp.forest_loss_driver.monocrop_classifier.tag_windows \
  --ds-path "$DS_PATH"
```

The script is idempotent and prints per-class-group tagged / already-tagged
counts.

## Train

The training transform samples an elapsed month uniformly from 1 through the
maximum available for that window and always sends 12 frames to
OlmoEarth-v1.2-Base. An elapsed month `m` gives the model `12-m` pre-loss frames
followed by the first `m` post-loss frames.

Fully frozen encoder:

```bash
rslearn model fit \
  --config data/forest_loss_driver/monocrop_classifier/model_frozen.yaml
```

Layer-wise learning-rate decay across the encoder:

```bash
rslearn model fit \
  --config data/forest_loss_driver/monocrop_classifier/model_llrd.yaml
```

Both configs request the six-month view for validation. If a window has fewer than
six post-loss months, the transform uses its maximum available month instead.
`test_config` intentionally reuses the validation split.

## Metrics

`model_llrd.yaml` and the experiment variants report per-window rather than
per-pixel metrics, defined in `metrics.py`. Each val/test sample is one full
128x128 window whose valid (non-nodata) pixels share a single class, so each
sample reduces to one prediction: the majority vote of the per-pixel argmax over
valid pixels (ties break toward the lowest class ID). The logged metrics are
`val_window_accuracy` / `test_window_accuracy` and a per-window confusion matrix
in wandb. Checkpointing monitors `val_window_accuracy`. `model_frozen.yaml`
still uses the original per-pixel metrics.

## Experiment variants

Three `model_llrd.yaml` variants change the training population or the class
space. All share the same dataset, split assignment, and per-window metrics:

| Config | Windows | Classes |
|---|---|---|
| `model_llrd_bolivia.yaml` | `groups: [bolivia]` | all 8 |
| `model_llrd_bolivia_soy.yaml` | `groups: [bolivia]` + `class_group: soy` tag | 3: nodata, mennonites_soybean, soybean via `class_id_mapping: {2: 1, 7: 2}` |
| `model_llrd_merged_soy.yaml` | all countries | 7: soybean merged into the mennonites_soybean slot via `class_id_mapping: {7: 2}` |

`model_llrd_bolivia_soy.yaml` requires the tagging step above. Class remapping
happens at train/eval time in `SegmentationTask.process_inputs`, so the label
rasters on disk keep the original 8-class IDs; note that prediction outputs from
the remapped configs are in the remapped class space.

```bash
rslearn model fit \
  --config data/forest_loss_driver/monocrop_classifier/model_llrd_bolivia.yaml
rslearn model fit \
  --config data/forest_loss_driver/monocrop_classifier/model_llrd_bolivia_soy.yaml
rslearn model fit \
  --config data/forest_loss_driver/monocrop_classifier/model_llrd_merged_soy.yaml
```

## Test elapsed months

Set `MONOCROP_NUM_POST_MONTHS` from 1 through 12. It defaults to 6 when unset. A
request above a window's available horizon is clamped for that window; for example,
an eight-month test uses six months on a window whose maximum is six.

```bash
MONOCROP_NUM_POST_MONTHS=1 rslearn model test \
  --config data/forest_loss_driver/monocrop_classifier/model_llrd.yaml \
  --ckpt_path /path/to/checkpoint.ckpt

MONOCROP_NUM_POST_MONTHS=12 rslearn model test \
  --config data/forest_loss_driver/monocrop_classifier/model_llrd.yaml \
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
DS_PATH=/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/monocrop_classifier_predict_20260721/
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
  --config data/forest_loss_driver/monocrop_classifier/model_llrd.yaml \
  --data.init_args.path "$DS_PATH"
```

The `RslearnWriter` callback writes the argmax class raster to the `output` layer.
