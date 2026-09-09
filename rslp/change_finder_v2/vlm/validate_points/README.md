# validate_points

Gauge how accurately Gemini can validate points that the land-cover-change (LCC) model
flagged as having a long-term change. Each flagged point is shown to the VLM as a short
time series of imagery (pulled from the VLM image database in `../image_db`), and the
model decides whether the point reflects a genuine, persistent land-cover change
(`positive`) or just seasonal/phenological variation, clouds, or artifacts (`negative`).

## Pipeline

```
v2 annotation JSON
        │  add_points.py            (creates windows + writes a PointSet)
        ▼
   points.json  ──►  rslearn prepare / materialize   (on the image database)
        │  run_gemini.py            (one S2 + up to one high-res image per year)
        ▼
 predictions.json
        │  compute_accuracy.py      (evaluation mode only)
        ▼
   metrics printed
```

## Modes

`add_points.py` runs in one of two modes:

- **evaluation** — for quantitatively measuring Gemini's accuracy. Includes labeled
  positive points (those annotated with `pre_change`, `post_change`,
  `first_date_change_noticeable`, and both `pre_category`/`post_category`) and negative
  points, storing the ground-truth label. The key year is the midpoint of the
  pre-change and post-change dates (for negatives, the midpoint of the entry's
  `time_range`).
- **deployment** — for running over real LCC-model output. Validates every point and
  errors out if any negative points exist, or if any point has `post_change` or
  `first_date_change_noticeable` set. Each point must provide `pre_change` (the
  predicted change date) and both predicted categories. The key year is the
  `pre_change` year.

## Usage

1. Add points from a v2 annotation file (creates windows in the image database and
   writes a point set):

   ```bash
   python -m rslp.change_finder_v2.vlm.validate_points.add_points \
       --annotations annotations.json \
       --image-db-path /path/to/image_db \
       --output points.json \
       --mode evaluation
   ```

2. Materialize the imagery with rslearn. The image database root needs a `config.json`
   first; copy the reference one from
   `data/change_finder_v2/vlm/image_db/config.json`:

   ```bash
   cp data/change_finder_v2/vlm/image_db/config.json /path/to/image_db/config.json
   rslearn dataset prepare --root /path/to/image_db --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5
   rslearn dataset materialize --root /path/to/image_db --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5 --no-use-initial-job
   ```

3. Run Gemini over the points:

   ```bash
   python -m rslp.change_finder_v2.vlm.validate_points.run_gemini \
       --points points.json \
       --output predictions.json
   ```

4. Compute accuracy (evaluation mode only):

   ```bash
   python -m rslp.change_finder_v2.vlm.validate_points.compute_accuracy \
       --predictions predictions.json
   ```

## Notes

- `run_gemini.py` selects one Sentinel-2 image and up to one high-resolution aerial
  image per calendar year (each nearest to mid-year), captions every image with its
  capture date, and passes them in chronological order. Layer names default to
  `sentinel2` and `esri`; override with `--s2-layer` / `--highres-layer` if your
  `config.json` uses different names.
- `compute_accuracy.py` only reports metrics in evaluation mode; deployment-mode
  predictions have no ground-truth labels and are skipped.
- The Gemini client uses Vertex AI (`--project`, `--location`, `--model`), defaulting
  to `earthsystem-dev-c3po` / `global` / `gemini-2.5-pro`.

## Files

- `add_points.py` — read a v2 annotation file, create image-database windows, write a `PointSet`.
- `run_gemini.py` — gather imagery per point, prompt Gemini, write a `PredictionSet`.
- `compute_accuracy.py` — accuracy, precision/recall/F1, and confusion matrix.
- `schema.py` — `PointRecord`, `PointSet`, `PointPrediction`, `PredictionSet`.
- `prompt.py` — image captioning (`label_image`) and prompt text (`build_validation_prompt`).
- `gemini.py` — `GeminiValidator`, the Vertex AI structured-output client.
