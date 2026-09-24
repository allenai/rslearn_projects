This dataset and model configuration rebuilds the forest loss driver training data
directly from the labels in OlmoEarth Studio, and switches the windows from Web
Mercator (9.55 m/pixel) to 10 m/pixel UTM (using the UTM zone appropriate for each
forest loss event). It also incorporates the new "Validatetest" labels, where ACA
validated the outputs of the deployed model across Brazil, Peru, Bolivia, Colombia,
and Ecuador.

The model configuration is otherwise the same as `20260401_peru_phase2/config.yaml`.

Note that the deployed inference pipeline in `olmoearth_projects` still creates Web
Mercator windows; it will need to be updated to UTM to match this dataset.


Source Studio Projects
----------------------

The projects are registered in `rslp/forest_loss_driver/create_dataset.py`
(`PROJECTS`). Adding a project means adding an entry there and re-running the script.

| Project | ID | Label field / hierarchy | Dataset group(s) |
| --- | --- | --- | --- |
| Forest Loss Driver Peru 7 | `25dfbe5f-4349-4646-b250-508afb2d42ba` | `tag_name`, original Peru hierarchy | `nadia2`, `nadia3`, `peru3`, `peru3_flagged_in_peru`, `brazil_interesting`, `peru_interesting` |
| Forest Loss Driver Brazil 7 | `f56e41c6-83ab-4a7f-9b14-443391f9b2ba` | `tag_name`, ACA hierarchy | `20250428_brazil_phase1` |
| Forest Loss Driver Colombia 7 | `a493cba0-466f-4604-8359-c437b78f7009` | `tag_name`, ACA hierarchy | `20250428_colombia_phase1` |
| Forest Loss Driver Brazil 12 | `f8137f81-15ac-4f94-b0fd-8ce5f62a6f78` | `tag_name`, ACA hierarchy | `20250428_brazil_phase2` |
| Forest Loss Driver Colombia 12 | `7732a6c0-cea0-46a5-9498-5d93eed51364` | `tag_name`, ACA hierarchy | `20250428_colombia_phase2` |
| Forest Loss Driver Peru 20260112c | `077f374b-2a72-45f4-9945-0290cc311201` | `tag_name`, ACA hierarchy | `20260112_peru` |
| Validatetest | `1d3727e6-5844-4e89-87f1-c1147417e180` | `validate`, flat categories | `20260821_validatetest` |

The first six projects were imported into Studio from rslearn datasets, so their tasks
carry `group` and `window` attributes naming the window in the legacy `combined`
dataset. The dataset groups above match the groups in that dataset (Peru 7 uses
attribute groups `brazil` and `peru` for `brazil_interesting` and `peru_interesting`).

The Validatetest tasks were sampled from the deployed model's outputs (Jul 2025 - Jun
2026, stratified by country and predicted category, see the `estrato` field). The
labeled category is in the `validate` metadata field; the `category` field is the
model prediction and is stored in the label feature as `category` for reference.


Labels
------

All labels are mapped to the ten flat categories that the model predicts:
`agriculture`, `mining`, `airstrip`, `road`, `logging`, `burned`, `landslide`,
`hurricane`, `river`, `none`.

- Original Peru hierarchy: `agriculture-generic`, `agriculture-small`,
  `agriculture-mennonite`, `agriculture-rice`, and `coca` map to `agriculture`;
  `flood` maps to `river`. `unknown`, `natural`, `human`, and `unlabeled` are skipped.
- ACA hierarchy: see `ACA_LABEL_MAP` in the script. `Natural_-_Unknown`,
  `Anthropic_-_Unknown`, `General_deforestation_Clearing`, `Mining_in_River`,
  `Urbanization_settlements`, `unknown`, and `unlabeled` are skipped (as they were in
  the legacy dataset).
- Validatetest: labels already use the flat categories.

Tasks with a skipped label, no annotation, or a `rejected` annotation do not get a
window. If a task has several annotations, the most recently updated one is used.
Confidence is recorded in the label feature but not filtered on.

Each window is named after the Studio task ID and has a single `label` feature
containing the forest loss polygon (falling back to the task polygon when the
annotation geometry is a point) with `new_label`, `raw_label`, `confidence`, the
Studio project/task/annotation IDs, `annotation_updated_time`, and the legacy
group/window names.


Split
-----

The split reproduces the one used by the legacy `combined` dataset so the validation
set stays comparable with `20260401_peru_phase2`:

- `20250428_brazil_phase1`, `20250428_brazil_phase2`, `20250428_colombia_phase1`,
  `20250428_colombia_phase2`, `peru3_flagged_in_peru`, `peru_interesting`: `val` if
  the first hex digit of the SHA-256 of the legacy window name is 0-3, else `train`.
- `peru3`, `nadia2`, `nadia3`, `brazil_interesting`, `20260112_peru`,
  `20260821_validatetest`: `train`.


Usage
-----

Set `STUDIO_API_KEY` and run (the script copies `config.json` from this directory into
the dataset if it does not exist):

```
DS=/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/20260924_utm/
python -m rslp.forest_loss_driver.create_dataset --ds-path $DS
rslearn dataset prepare --root $DS --workers 64 --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root $DS --workers 64 --ignore-errors --retry-max-attempts 5 --retry-backoff-seconds 5
```

Use `--dry-run` to only print the per-project selection summary, and `--project-id`
(repeatable) to restrict to a subset of the registered projects. Re-running the script
is incremental: existing windows are left alone unless their Studio annotation has
been updated since, in which case the label layer is rewritten.

Then train the model, with and without the new Validatetest labels:

```
rslearn model fit --config data/forest_loss_driver/20260924_utm/config.yaml
rslearn model fit --config data/forest_loss_driver/20260924_utm/config_no_new_labels.yaml
```


Caveats
-------

- The dataset config materializes the four least cloudy Sentinel-2 images in the 180
  days before and after each event (`max_matches: 4`), which is exactly what the model
  reads with `load_all_layers: true`, so windows where fewer than four images are
  available on either side are excluded automatically. This will affect some of the most recent
  (Apr-Jun 2026) Validatetest events.
- Validation metrics will differ slightly from `20260401_peru_phase2` due to the UTM
  re-projection and the switch from Planetary Computer to the OlmoEarth Datasets data
  source (`olmoearth_run...olmoearth_datasets.sentinel2_l2a.Sentinel2L2A`, as used by
  the deployed inference pipeline), so
  compare `config.yaml` against `config_no_new_labels.yaml` rather than against the
  older runs.
- The Validatetest labels are single-annotator validations of the model's own
  predictions, stratified by predicted category, so they are biased toward what the
  model already predicts. They are used for training only.
