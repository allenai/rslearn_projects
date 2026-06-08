# Annotation Timestamp Helper

This helper predicts timestamp fields for `change_finder_v2` annotation JSONs:

- `pre_change`
- `first_date_change_noticeable`
- `post_change`

It is meant for point-level annotation assistance, not large-area mapping. Each
rslearn window is a 32x32 chip centered on the first positive point in an
annotation entry.

## Input Requirements

Each annotation entry must have:

- `time_range` with a start and end timestamp that can support at least 60
  30-day Sentinel-2 periods, e.g.
  `["2020-09-09", "2026-09-09"]`
- at least one item in `positive_points`

Only `positive_points[0]` is used. For training JSONs, that point must have all
three timestamp fields filled; incomplete training entries are skipped. Inference
entries only need the point and a long-enough `time_range`.

The dataset config requests up to 84 Sentinel-2 mosaics using 30-day periods.
Training randomly crops a five-year, 60-frame subsequence. Inference always uses
the middle 60-frame crop. If fewer than 60 frames are prepared, the transform
repeats the final real frame and continues the 30-day timestamp cadence until the
model input reaches 60 frames. Labels that fall in this padded tail are marked
invalid for loss/metrics.

## 1. Create the Dataset

Set the OlmoEarth Datasets credentials:

```bash
export OEDATASETS_API_URL=https://datasets.olmoearth.allenai.org
export DATASETS_API_TOKEN=<your-token>
```

Create the rslearn dataset root and copy the dataset config:

```bash
DS=/path/to/annotation_timestamp_helper_dataset
mkdir -p "$DS"
cp data/change_finder_v2/annotation_timestamp_helper/config.json "$DS/config.json"
```

Create windows, labels, and the merge manifest:

```bash
python -m rslp.change_finder_v2.annotation_timestamp_helper.create_windows \
    --train-json annotated_train_a.json \
    --train-json annotated_train_b.json \
    --infer-json inference_needs_timestamps.json \
    "$DS"
```

The script writes:

- `train`, `val`, and `inference` window groups
- `label_timestamps` vector labels for train/val windows
- `timestamp_helper_manifest.json`, used later for merging predictions

## 2. Prepare And Materialize Imagery

Run rslearn's standard prepare/materialize flow:

```bash
rslearn dataset prepare --root "$DS" --workers 32
rslearn dataset materialize --root "$DS" --workers 128
```

## 3. Train

```bash
rslearn model fit \
    --config data/change_finder_v2/annotation_timestamp_helper/config.yaml \
    --data.init_args.path="$DS"
```

The model runs five OlmoEarth passes over the cropped five-year sequence, uses
the center spatial token from each monthly frame, and predicts three 60-way
classification heads.

## 4. Predict On Inference Windows

Run prediction with the trained checkpoint:

```bash
rslearn model predict \
    --config data/change_finder_v2/annotation_timestamp_helper/config_predict.yaml \
    --data.init_args.path="$DS" \
    --ckpt_path /path/to/best.ckpt
```

Predictions are written to the `output_timestamps` vector layer. Each feature
contains monotonic-decoded `pre_idx`/`pre_date`, `first_idx`/`first_date`, and
`post_idx`/`post_date` properties, plus probability arrays for review.

## 5. Merge Predictions Into A New JSON File

Write an updated copy of one inference annotation JSON:

```bash
python -m rslp.change_finder_v2.annotation_timestamp_helper.merge_predictions \
    --dataset-path "$DS" \
    --json inference_needs_timestamps.json \
    --output-json inference_with_predicted_timestamps.json
```

For each inference point, the merge script:

- reads the decoded timestamp date properties from `output_timestamps`
- overwrites the three timestamp fields in the output JSON copy

The source JSON file is not modified. Run the command once per inference JSON if
you created windows from multiple inference files.
