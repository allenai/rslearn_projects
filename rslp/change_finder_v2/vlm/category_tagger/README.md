# category_tagger

Use a VLM (Gemini) to assign fine-grained change categories to points that the
land-cover-change (LCC) model flagged as having a long-term change. Each point is shown
to the model as a short time series of imagery (pulled from the VLM image database in
`../image_db`), and the model assigns one or more change categories.

Only fully-labeled positive points are considered (those with `pre_change`,
`post_change`, `first_date_change_noticeable`, `pre_category`, and `post_category` all
set); negative points and partially-labeled positives are skipped.

`add_points.py` has two modes:

- `all` (default): add every fully-labeled positive point. Every point is an *unlabeled*
  change for the model to categorize.
- `evaluation`: only add points that also have at least one hand-labeled ground-truth
  change category (`pre_change_category`, `post_change_category`, or
  `same_change_category`). Use this to quantitatively evaluate Gemini's accuracy with
  `compute_accuracy.py`.

Ground-truth change categories are always recorded on the point when present (regardless
of mode), so they flow through `run_gemini.py` into `categories.json` and can be scored.
These ground-truth fields share the same names and value sets as the model's predictions
(see Categories below); they are distinct from the coarse `pre_category` / `post_category`
land-cover classes that only prime the prompt.

## Categories

Categories fall into three groups:

- **Pre-class** (named for what was lost / removed): `deforestation`, `urban_erosion`,
  `removed_crop_structure`, `wetland_loss`, `water_contract`.
- **Post-class** (named for what appeared): `vegetation_growth`, `new_building`,
  `new_road`, `new_infrastructure`, `new_crop_field`, `new_crop_structure`,
  `new_aquafarm`, `site_clearing`, `water_expand`, `mining`.
- **Same-class** (the land cover class stays the same with variation):
  `agricultural_activity`, `wildfire`.

The model picks one pre-class category and/or one post-class category, OR a single
same-class category. The prompt also suggests the most likely categories for the
point's coarse land-cover transition (e.g. `tree -> urban/built-up` suggests
`deforestation` plus `new_building`/`new_road`/...). If none of the categories fit, the
model sets `flagged_for_review` so the point can be reviewed by hand.

## Pipeline

```
v2 annotation JSON
        │  add_points.py            (creates windows + writes a PointSet)
        ▼
   points.json  ──►  rslearn prepare / materialize   (on the image database)
        │  run_gemini.py            (one S2 + up to one high-res image per year)
        ▼
 categories.json
        │  compute_accuracy.py      (evaluation mode only; needs ground truth)
        ▼
    metrics
```

## Usage

1. Add points from a v2 annotation file (creates windows in the image database and
   writes a point set):

   ```bash
   python -m rslp.change_finder_v2.vlm.category_tagger.add_points \
       --annotations annotations.json \
       --image-db-path /path/to/image_db \
       --output points.json \
       --mode all          # or: --mode evaluation
   ```

2. Materialize the imagery with rslearn. The image database root needs a `config.json`
   first; copy the reference one from
   `data/change_finder_v2/vlm/image_db/config.json`:

   ```bash
   cp data/change_finder_v2/vlm/image_db/config.json /path/to/image_db/config.json
   rslearn dataset prepare --root /path/to/image_db --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5 --enabled-layers sentinel2
   rslearn dataset prepare --root /path/to/image_db --workers 4 --retry-max-attempts 5 --retry-backoff-seconds 5 --enabled-layers esri
   rslearn dataset materialize --root /path/to/image_db --workers 128 --retry-max-attempts 5 --retry-backoff-seconds 5 --no-use-initial-job --enabled-layers sentinel2
   rslearn dataset materialize --root /path/to/image_db --workers 4 --retry-max-attempts 5 --retry-backoff-seconds 5 --no-use-initial-job --enabled-layers esri
   ```

3. Run Gemini over the points:

   ```bash
   python -m rslp.change_finder_v2.vlm.category_tagger.run_gemini \
       --points points.json \
       --output categories.json
   ```

4. (Evaluation only) Score the predictions against the ground truth:

   ```bash
   python -m rslp.change_finder_v2.vlm.category_tagger.compute_accuracy \
       --predictions categories.json
   ```

   This reports two metrics over the labeled subset:

   - **Exact accuracy**: the `(pre, post, same)` change categories all match exactly.
   - **Mostly-right accuracy**: at least one non-null change category matches in the
     same slot (so a same-class label must be predicted exactly right, while a pre/post
     label counts if at least one of the two sides is correct).

## Notes

- The key year for each point is the midpoint year of its `pre_change` and `post_change`
  dates.
- `run_gemini.py` samples imagery relative to each point's change dates, in three
  segments: the 2 years before `pre_change`, the interval between `pre_change` and
  `post_change`, and the 2 years after `post_change`. For Sentinel-2, each segment is
  split into 4 even periods (so ~6-month periods before/after) and the least-cloudy
  image (a center-clarity heuristic) is taken per period (up to ~12 total). For the
  high-resolution aerial layer, up to 4 images are kept per segment, spread out in time
  via farthest-point sampling so closely-spaced releases are dropped. Each image is
  captioned beneath it with its type and capture date and a magenta box is drawn around
  the center; they are passed in chronological order. The `pre_change`,
  `first_observable`, and `post_change` dates are also stated in the prompt. Layer names
  default to `sentinel2` and `esri`; override with `--s2-layer` / `--highres-layer` if
  your `config.json` uses different names.
- The Gemini client uses Vertex AI (`--project`, `--location`, `--model`), defaulting
  to `earthsystem-dev-c3po` / `global` / `gemini-2.5-pro`.

## Files

- `add_points.py` — read a v2 annotation file, create image-database windows, write a `PointSet` (`all` or `evaluation` mode).
- `run_gemini.py` — gather imagery per point, prompt Gemini, write a `CategorySet`.
- `compute_accuracy.py` — score a `CategorySet` against ground truth (exact + mostly-right accuracy).
- `schema.py` — `PointRecord`, `PointSet`, `CategoryPrediction`, `CategorySet`.
- `prompt.py` — category definitions, image captioning (`label_image`), and prompt text (`build_category_prompt`).
- `gemini.py` — `GeminiCategorizer`, the Vertex AI structured-output client.
