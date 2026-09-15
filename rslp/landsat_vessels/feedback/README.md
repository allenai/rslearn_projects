# Landsat vessel feedback loop

Once a model is deployed to Skylight integration, wait for it to run and for users to
flag false positives, then fold that feedback back into the classifier: pull the
feedback, turn it into training windows, retrain, validate, and republish the served
Docker image — so the deployed detector keeps improving. Repeat each time enough new
feedback has accumulated.

This is the standing version of the one-off round-0 flow
(`/weka/dfive-default/yawenz/landsat/README.md`, group `feedback_20260325`) and the
round-1 annotation flow (`rslp/landsat_vessels/annotations/`).

## Current state (update every cycle)

- **Deployed:** `landsat_vessels_v1.0.0` = run `olmoearth_base_layerdecay_20260908d`,
  config `data/landsat_vessels/config_classifier_20260908d.yaml`, operating point
  detector `0.7` / classifier `0.99`, on **integration** (<https://app-int.skylight.earth>).
- **Last feedback batch:** `feedback_20260911` — 28 v1.0.0 false positives, materialized
  into group `feedback_20260911`.
- **Latest retrain (NOT yet published):** run `olmoearth_base_layerdecay_20260911`,
  config `data/landsat_vessels/config_classifier_20260911.yaml`, best
  `val_correct_f1 ≈ 0.884` (wandb `landsat_vessel_classification_v2`, run `nvxvka3r`).
  QA vs baseline: kept detections 116→70 (feedback scenes `109027` 42→2, `109029` 16→4,
  `101068` 3→0; non-feedback scenes stable). **Next:** publish if approved (§5).

## Prereqs

- Repo `/weka/dfive-default/yawenz/rslearn_projects`, branch
  `yawenz/20260602-landset-vessel-classifier-v2`.
- venv `/weka/dfive-default/yawenz/.venv-rslearn` — the CLI is `.venv-rslearn/bin/rslearn`
  (there is **no** `python -m rslearn`); Python is `.venv-rslearn/bin/python`.
- Env: `RSLP_PREFIX=/weka/dfive-default/rslearn-eai`, `PYTHONPATH=$PWD`,
  `CUDA_VISIBLE_DEVICES=0`, and `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (Landsat
  scenes are read from AWS S3 — ask the user; never commit or print them).
- `gcloud`/`gsutil` authenticated as `yawenz@allenai.org`. Use a GPU node (1 GPU is enough)
  and a per-batch working dir, e.g. `/weka/dfive-default/yawenz/landsat/feedback_<date>/`.

## The loop

```
Skylight in-app feedback ─(1)pull─▶ feedback_<date>.csv ─(2)create_windows─▶ rslearn windows
   ─(3)add_to_training─▶ config_classifier_<date>.yaml ─(4)train─▶ best.ckpt
   ─(4.5)validate (run_recent_inference, new vs baseline) ─(5)publish─▶ Docker + Skylight
```

Each cycle is its own dated group `feedback_<date>` and its own dated config/run
`..._<date>` — a separate, auditable slice of the training data.

### 1. Pull + enrich + filter feedback

Export the in-app feedback CSV from the admin panel
(`.../admin?selected-tab=in-app-feedback`) → `in_app_feedback.csv` (columns
`event_id, username, value, timestamp, ...`; no coordinates). The Skylight API is
**IP-blocked from compute** (integration returns `403`, production `401` at the load
balancer), so coordinates are enriched **offline from the sat-service detection bucket on
GCS** — the authoritative source, which also records each detection's
`rslearn_model_version`.

```bash
python -m rslp.landsat_vessels.feedback.pull \
  --input  /weka/dfive-default/yawenz/landsat/feedback_<date>/in_app_feedback.csv \
  --out    /weka/dfive-default/yawenz/landsat/feedback_<date>/feedback_<date>.csv \
  --environment integration --source gcs \
  --username yawenz@allenai.org \
  --since <YYYYMMDD-of-current-deploy> --date-field submission \
  --keep bad_only \
  --model-version <DEPLOYED_MODEL_VERSION>     # e.g. landsat_vessels_v1.0.0
```

Filters (cheap ones first, so GCS is only read for survivors):
- `--username` — exact allowlist (repeatable); overrides the trusted-domain list in `config.py`.
- `--keep bad_only` — false positives only (`BAD → incorrect`); `good_bad` also keeps `GOOD → correct`.
- `--since` / `--until` with `--date-field` — `acquisition` (the date in the product id,
  = when the detection was produced), `submission` (when the feedback was filed), or `event_time`.
- `--model-version` — keep only detections from the model under evaluation (substring
  match on the detection JSON's `rslearn_model_version`), applied after GCS enrichment.
  **Critical:** integration can serve a mix of versions; only fold in the FPs the model
  you monitor produced.

Enrichment sources (`--source`): `gcs` (default here, offline, authoritative), `api`
(Skylight GraphQL, needs `--token`, only from an allowlisted network), `tile`
(coordinates already in the export). Buckets per environment are in `config.py`.

### 2. Create + materialize windows

```bash
export PATH=/weka/dfive-default/yawenz/.venv-rslearn/bin:$PATH   # so create_windows can shell out to `rslearn`
python -m rslp.landsat_vessels.feedback.create_windows \
  --csv   /weka/dfive-default/yawenz/landsat/feedback_<date>/feedback_<date>.csv \
  --group feedback_<date> \
  --materialize --workers 32
```

One 64 px @ 15 m window per feedback row in group `feedback_<date>`, each with a `label`
layer (`label=incorrect`, `model_version`, `annotator`). When the CSV carries a Landsat
`scene_id` the exact product is pinned; if it is gone, another product of the same
acquisition is substituted **RT → T1 → T2** (production runs on RT, so RT is the imagery
the deployed model saw). Windows with no `scene_id` are matched by time. `--materialize`
runs `rslearn dataset materialize` (drop it to just write windows and print the commands).

### 3. Generate the training config

```bash
python -m rslp.landsat_vessels.feedback.add_to_training --group feedback_<date>
```

Writes `data/landsat_vessels/config_classifier_<date>.yaml` = the deployed config +
`feedback_<date>` added to `data.init_args.default_config.groups` +
`run_name: olmoearth_base_layerdecay_<date>`. The only diff is the added group and the run
name; val stays `feedback_20260325`, test stays the frozen `round1_20260803` set, so every
cycle's model is comparable. **Do not edit other configs.**

### 4. Retrain (1 GPU)

Train locally with the rslearn CLI (what `20260911` used; the beaker `launch_finetune`
path is missing from this checkout):

```bash
export RSLP_PREFIX=/weka/dfive-default/rslearn-eai CUDA_VISIBLE_DEVICES=0
export AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret>
nohup /weka/dfive-default/yawenz/.venv-rslearn/bin/rslearn model fit \
  --config data/landsat_vessels/config_classifier_<date>.yaml \
  > /weka/dfive-default/yawenz/landsat/feedback_<date>/train_<date>.log 2>&1 &
echo $! > /weka/dfive-default/yawenz/landsat/feedback_<date>/train.pid
```

Best checkpoint (metric `val_correct_f1`) lands at
`${RSLP_PREFIX}/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_<date>/best.ckpt`.
**Healthy start** = the log advances past `Matplotlib is building the font cache` to
dataloaders/`Epoch 0` and `nvidia-smi` shows GPU memory > 0.

### 4.5 Validate before publishing (QA gate)

Re-run the recent-inference QA over the fixed scene list with the **new** classifier and
compare against the deployed baseline. The classifier config's `run_name` resolves the
checkpoint, so just point `--classify_config` at the new dated config.

```bash
export AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret>
export RSLP_PREFIX=/weka/dfive-default/rslearn-eai PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
python -m rslp.landsat_vessels.scripts.run_recent_inference \
  --scene_csv /weka/dfive-default/yawenz/landsat/20260911_results/scene.csv \
  --out_dir   /weka/dfive-default/yawenz/landsat/<date>_results/new_classifier_<date> \
  --classify_config data/landsat_vessels/config_classifier_<date>.yaml \
  --det_thr 0.7 --cls_thr 0.99 --workers 3
```

Diff `detections.csv` kept counts and eyeball `figures/<scene>/candidates_grid.png`
new-vs-baseline: FP-heavy scenes should drop sharply, non-feedback scenes stay stable.
`run_recent_inference` writes the exact `patched_classifier.yaml` it used — check its
`run_name` to confirm the right checkpoint loaded.

### 5. Publish the new version

`publish.py` plans by default (prints every action, touches nothing); add flags to apply.

```bash
python -m rslp.landsat_vessels.feedback.publish \
  --run-name olmoearth_base_layerdecay_<date> \
  --config   data/landsat_vessels/config_classifier_<date>.yaml \
  --old-run-name <run currently in the Dockerfile>   # e.g. olmoearth_base_layerdecay_20260908d
  --upload --patch            # then --build --push (or run the printed docker commands)
```

1. **upload** `best.ckpt` →
   `gs://ai2-rslearn-projects-data/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_<date>/best.ckpt`.
2. **patch** `Dockerfile` (classifier `wget` `<old_run>` → `<new_run>`) and `config.py`
   (`CLASSIFY_MODEL_CONFIG` → the new dated config).
3. **build & push** the `landsat_vessels:latest` image (needs a docker daemon + registry
   creds; usually run by hand from the printed commands). Pass `--image
   landsat_vessels:v1.1.1` to tag the version.

Then **smoke-test the container endpoint** (port 5555), not just the local checkout,
before redeploying to Skylight.

The `v1.x.x` label is **not** in this repo — it is assigned at deploy time on the
Skylight/sat-service side plus the docker image tag. Coordinate the redeploy + version
label with the Skylight deploy owner.

### 6. Close the cycle

So the next loop targets the newly-deployed model, update `config.py`
(`DEPLOYED_MODEL_VERSION`, `DEPLOYED_RUN_NAME`, `BASE_CLASSIFIER_CONFIG`) and the
**Current state** section above.
