# Landsat vessel feedback loop

Turn false positives flagged in the Skylight platform into new classifier training data,
retrain, and republish the served Docker image so the deployed model keeps improving.

> **Running the loop / picking it up next month?** Start with **`RUNBOOK.md`** in this
> directory — it has the step-by-step procedure, the current deployed state, the commands
> that actually work on this infra, and the publish/versioning steps. This README is the
> per-tool flag reference.

This is the standing, repeatable version of the one-off round-0 flow documented in
`/weka/dfive-default/yawenz/landsat/README.md` (`enrich_feedback.py` +
`create_feedback_windows.py`, group `feedback_20260325`) and the round-1 annotation flow
in `rslp/landsat_vessels/annotations/`. The `v1.0.0` model
(`olmoearth_base_layerdecay_20260908d`) is deployed to the **integration** environment
(<https://app-int.skylight.earth>); as false positives are flagged there each month, run
this loop to fold them in.

## The loop

```
Skylight in-app feedback
      │  (1) pull.py            pull + enrich + filter -> feedback_<date>.csv
      ▼
normalised CSV
      │  (2) create_windows.py  CSV -> rslearn windows (label layer) in group feedback_<date>
      ▼
classifier dataset  (dataset_20250624)
      │  (3) add_to_training.py adds the group to the classifier config -> config_classifier_<date>.yaml
      ▼
(4) train            python -m rslp.main olmoearth_pretrain launch_finetune ...  (1 GPU)
      │
      ▼
best.ckpt on weka
      │  (5) publish.py         upload to GCS + patch Dockerfile/config + build & push image
      ▼
redeploy to Skylight
```

Each month gets its own dated group, e.g. `feedback_20260911`, so every batch is a
separate, auditable slice of the training data — exactly like `feedback_20260325`.

## Step by step (example: `feedback_20260911`)

Assumes `RSLP_PREFIX=/weka/dfive-default/rslearn-eai`, AWS creds exported (for the AWS
Landsat data source), and the rslearn env active. Pick a working dir per batch, e.g.
`/weka/dfive-default/yawenz/landsat/feedback_20260911/`.

### 1. Pull feedback from Skylight

Export the in-app feedback CSV from the integration admin panel
(<https://app-int.skylight.earth/admin?selected-tab=in-app-feedback>) → save as
`in_app_feedback.csv` (columns: `event_id, event_type, username, value, timestamp, ...`;
no coordinates). The Skylight integration API is **IP-restricted and unreachable from
compute** (the load balancer returns `403` regardless of token), so coordinates are
enriched **offline from the sat-service detection bucket on GCS** — the authoritative
source, which also records each detection's `rslearn_model_version`.

```bash
python -m rslp.landsat_vessels.feedback.pull \
  --input  /weka/dfive-default/yawenz/landsat/feedback_20260911/in_app_feedback.csv \
  --out    /weka/dfive-default/yawenz/landsat/feedback_20260911/feedback_20260911.csv \
  --environment integration --source gcs \
  --username yawenz@allenai.org \
  --since 20260911 --date-field submission \
  --keep bad_only \
  --model-version landsat_vessels_v1.0.0
```

Filters, in order (cheap ones first, so GCS is only read for survivors):
- `--username` — exact allowlist (repeatable); overrides the trusted-domain list in `config.py`.
- `--keep bad_only` — false positives only (`BAD → incorrect`); `good_bad` also keeps `GOOD → correct`.
- `--since` / `--until` with `--date-field` — `acquisition` (default, the date in the product id
  = when the detection was produced), `submission` (when the feedback was filed), or `event_time`.
- `--model-version` — keep only detections produced by the model under evaluation
  (substring match on the detection JSON's `rslearn_model_version`); applied after GCS
  enrichment. Integration can serve a mix of versions, so this matters.

Enrichment sources (`--source`): `gcs` (default here, offline, authoritative),
`api` (Skylight GraphQL, needs `--token`, only works from an allowlisted network),
`tile` (coordinates already in the export). The GCS bucket per environment is in
`config.py` (`SKYLIGHT_DETECTION_BUCKETS`).

### 2. Create rslearn windows

```bash
python -m rslp.landsat_vessels.feedback.create_windows \
  --csv   /weka/dfive-default/yawenz/landsat/feedback_20260911/feedback_20260911.csv \
  --group feedback_20260911 \
  --materialize --workers 32
```

One 64 px @ 15 m window per feedback row in group `feedback_20260911`, each with a
`label` layer. When the CSV carries a Landsat `scene_id` the exact product is pinned;
if that product is gone, it substitutes another product of the same acquisition **RT
first, then T1, then T2** (production runs on RT, so RT is the imagery the deployed model
actually saw). Windows with no `scene_id` are matched by time. `--materialize` runs
`rslearn dataset materialize` to populate the imagery (drop it to just write windows and
print the commands). All windows go to the `train` split by default (`--val-fraction`
holds some out).

### 3. Add the group to the training config

```bash
python -m rslp.landsat_vessels.feedback.add_to_training --group feedback_20260911
```

Writes `data/landsat_vessels/config_classifier_20260911.yaml` — the deployed Run-d config
plus `feedback_20260911` in `default_config.groups` and `run_name`
`olmoearth_base_layerdecay_20260911` (date suffix, no experiment letter). The only diff
from the deployed config is the added group and the run name; val stays
`feedback_20260325`. Prints the launch command.

### 4. Retrain (1 GPU)

```bash
python -m rslp.main olmoearth_pretrain launch_finetune \
  --olmoearth_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 \
  --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 \
  --config_paths+=data/landsat_vessels/config_classifier_20260911.yaml \
  --cluster+=ai2/ceres-cirrascale \
  --project_name landsat_vessel_classification_v2 \
  --run_name olmoearth_base_layerdecay_20260911 --gpus 1
```

Sanity-check the new model before publishing — the smoke test and threshold sweep in the
parent `README.md`, and the operating point (detector `0.7` / classifier `0.99`).

### 5. Publish the new Docker image

```bash
python -m rslp.landsat_vessels.feedback.publish \
  --run-name olmoearth_base_layerdecay_20260911 \
  --config data/landsat_vessels/config_classifier_20260911.yaml \
  --upload --patch          # then --build --push, or run the printed docker commands
```

Uploads `best.ckpt` to
`gs://ai2-rslearn-projects-data/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_20260911/best.ckpt`
(the run name carries the date), points the Dockerfile's classifier `wget` and
`config.py`'s `CLASSIFY_MODEL_CONFIG` at the new run, and builds/pushes the image.
Smoke-test the **container endpoint** (not just the local checkout) before redeploying to
Skylight. Run without flags first to see the plan.

## Files

| file | stage | role |
|---|---|---|
| `config.py` | — | shared constants: Skylight endpoints + detection buckets, trusted users, dataset root, project/run names, GCS paths |
| `pull.py` | 1 | filter Skylight in-app feedback (user/date/value/model-version) + enrich lat/lon from the GCS detection bucket → normalised CSV |
| `create_windows.py` | 2 | CSV → rslearn windows + `label` layers, pin/substitute scenes, materialize |
| `add_to_training.py` | 3 | generate a dated classifier config with the new group + run_name |
| `publish.py` | 5 | upload checkpoint to GCS, patch Dockerfile/config, build & push image |

## Notes

- **Environment.** Defaults target `integration` (where v1.0.0 runs). Pass
  `--environment production` to pull from <https://app.skylight.earth> instead.
- **Label semantics.** The classifier task is `correct` vs `incorrect`; a Skylight `BAD`
  (false positive) becomes an `incorrect` training target — the hard negative the loop
  exists to collect. `GOOD` becomes `correct`.
- **Model version.** Integration can serve more than one model version at once, so filter
  feedback to the model you are monitoring with `--model-version` (from the detection
  JSON's `rslearn_model_version`); otherwise you would train v1.0.0 on v0.0.21's mistakes.
- **Coordinates come from GCS, not the API.** The Skylight integration API is IP-walled
  off from compute (`403` at the load balancer). `pull.py --source gcs` reads the
  authoritative sat-service detection JSONs instead, matching each feedback `event_id`'s
  trailing index to the detection's crop index.
- **Window size.** 64 px matches `feedback_20260325`; the classifier reads the centre
  32 px via `CenterCrop`. Mixed sizes across groups are fine (see the parent
  `annotations/README.md`); keep `crop_size: 64` while any 64 px group is in the run.
- **Evaluation stays fixed.** Feedback windows enter *training*; val is
  `feedback_20260325` and test is the frozen `round1_20260803` set, so each month's model
  is comparable to the last.
