# Landsat vessel feedback → retrain → publish RUNBOOK

**Start here** if you are an agent picking up the monthly Landsat-vessel feedback loop.
This is the step-by-step playbook (with the current state, the commands that actually
work on this infra, and the gotchas). For per-tool flag reference see `README.md` in this
directory; this file is the *procedure* + *state*.

The goal: each month, the false positives users flag in Skylight get folded back into the
classifier training set, the model is retrained, validated, and republished, so the
deployed detector keeps improving.

---

## Current state (update this section every cycle)

- **Deployed model:** `landsat_vessels_v1.0.0` = classifier run
  `olmoearth_base_layerdecay_20260908d`, config `data/landsat_vessels/config_classifier_20260908d.yaml`,
  operating point detector `0.7` / classifier `0.99`. Running on the **integration**
  environment (`https://app-int.skylight.earth`).
- **Last feedback batch:** `feedback_20260911` — 28 v1.0.0 false positives pulled +
  enriched + materialized into the classifier dataset (group `feedback_20260911`).
- **Latest retrain (NOT yet published):** run `olmoearth_base_layerdecay_20260911`,
  config `data/landsat_vessels/config_classifier_20260911.yaml`,
  `best.ckpt` at `${RSLP_PREFIX}/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_20260911/best.ckpt`,
  best `val_correct_f1 ≈ 0.884` (wandb project `landsat_vessel_classification_v2`,
  run id `nvxvka3r`).
- **Validation done:** ran `run_recent_inference` over the 6 QA scenes with the new
  classifier (`20260911_results/new_classifier_20260911/`) vs the v1.0.0 baseline
  (`20260911_results/`). Kept detections dropped 116→70 (−40%), concentrated on the
  feedback scenes (`109027` 42→2, `109029` 16→4, `101068` 3→0) with non-feedback scenes
  stable — i.e. the loop fixes the flagged FPs without hurting recall elsewhere.
- **Next action if approved:** publish this run as the next version (see §5). Nothing has
  been pushed to Docker/Skylight.

---

## Prereqs / environment

- Repo: `/weka/dfive-default/yawenz/rslearn_projects`, branch
  `yawenz/20260602-landset-vessel-classifier-v2`.
- venv: `/weka/dfive-default/yawenz/.venv-rslearn` — invoke the CLI as
  `.venv-rslearn/bin/rslearn` (there is **no** `python -m rslearn` entrypoint) and Python
  as `.venv-rslearn/bin/python`.
- Env vars: `RSLP_PREFIX=/weka/dfive-default/rslearn-eai`,
  `PYTHONPATH=/weka/dfive-default/yawenz/rslearn_projects`,
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (Landsat scenes are read from AWS S3 —
  ask the user for the keys; never commit/print them), and `CUDA_VISIBLE_DEVICES=0`.
- GCS: `gcloud`/`gsutil` authenticated as `yawenz@allenai.org` (reads the sky-int
  detection bucket and writes the projects bucket).
- Use a GPU node (1 GPU is enough for the retrain).
- Pick a per-batch working dir, e.g. `/weka/dfive-default/yawenz/landsat/feedback_<date>/`.

---

## The loop (one cycle)

```
Skylight in-app feedback ─(1)pull─▶ feedback_<date>.csv ─(2)create_windows─▶ rslearn windows
   ─(3)add_to_training─▶ config_classifier_<date>.yaml ─(4)train─▶ best.ckpt
   ─(4.5)validate (run_recent_inference, new vs baseline) ─(5)publish─▶ Docker + Skylight
```

Each month = its own dated group `feedback_<date>` and its own dated config/run
`..._<date>` (date suffix only, no experiment letter).

### 1. Pull + enrich + filter feedback

Export the in-app feedback CSV from the Skylight admin panel
(`.../admin?selected-tab=in-app-feedback`) → `in_app_feedback.csv`. The Skylight API is
**IP-blocked from compute**, so coordinates are enriched offline from the sat-service
detection bucket on GCS.

```bash
cd /weka/dfive-default/yawenz/rslearn_projects
python -m rslp.landsat_vessels.feedback.pull \
  --input  /weka/dfive-default/yawenz/landsat/feedback_<date>/in_app_feedback.csv \
  --out    /weka/dfive-default/yawenz/landsat/feedback_<date>/feedback_<date>.csv \
  --environment integration --source gcs \
  --username yawenz@allenai.org \
  --since <YYYYMMDD-of-current-deploy> --date-field submission \
  --keep bad_only \
  --model-version <DEPLOYED_MODEL_VERSION>     # e.g. landsat_vessels_v1.0.0
```

`--model-version` is critical: integration can serve a mix of versions; only fold in the
FPs produced by the model you are monitoring.

### 2. Create + materialize windows

```bash
export PATH=/weka/dfive-default/yawenz/.venv-rslearn/bin:$PATH   # so create_windows can shell out to `rslearn`
python -m rslp.landsat_vessels.feedback.create_windows \
  --csv   /weka/dfive-default/yawenz/landsat/feedback_<date>/feedback_<date>.csv \
  --group feedback_<date> \
  --materialize --workers 32
```

One 64 px @ 15 m window per FP in group `feedback_<date>`, each with a `label` layer
(`label=incorrect`, `model_version`, `annotator`). Scenes are pinned to the exact product
when present, else substituted **RT → T1 → T2** (production runs on RT).

### 3. Generate the training config

```bash
python -m rslp.landsat_vessels.feedback.add_to_training --group feedback_<date>
```

Writes `data/landsat_vessels/config_classifier_<date>.yaml` = the currently-deployed
config + `feedback_<date>` added to `data.init_args.default_config.groups` +
`run_name: olmoearth_base_layerdecay_<date>`. Val stays `feedback_20260325`, test stays
the frozen set — so every month's model is comparable. **Do not edit other configs.**

### 4. Retrain (1 GPU) — local `rslearn model fit`

Train locally with the rslearn CLI (what was used for `20260911`; the beaker
`launch_finetune` path is missing from this checkout):

```bash
cd /weka/dfive-default/yawenz/rslearn_projects
export RSLP_PREFIX=/weka/dfive-default/rslearn-eai CUDA_VISIBLE_DEVICES=0
export AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret>
nohup /weka/dfive-default/yawenz/.venv-rslearn/bin/rslearn model fit \
  --config data/landsat_vessels/config_classifier_<date>.yaml \
  > /weka/dfive-default/yawenz/landsat/feedback_<date>/train_<date>.log 2>&1 &
echo $! > /weka/dfive-default/yawenz/landsat/feedback_<date>/train.pid
```

- Checkpoints auto-manage via `management_dir/project_name/run_name`: best lands at
  `${RSLP_PREFIX}/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_<date>/best.ckpt`.
  Metric: `val_correct_f1` (monitored by `ManagedBestLastCheckpoint`). wandb project
  `landsat_vessel_classification_v2`, run `olmoearth_base_layerdecay_<date>`.
- **Healthy start** = log advances past `Matplotlib is building the font cache` to
  dataloaders/`Epoch 0`, and `nvidia-smi` shows GPU memory > 0. Watch best.ckpt mtime +
  the `saved best checkpoint (val_correct_f1=...)` lines.

### 4.5 Validate before publishing (QA gate)

Re-run the recent-inference QA over the fixed scene list with the **new** classifier and
compare kept-detection counts + `candidates_grid.png` figures against the deployed
baseline. Because the classifier config's `run_name` resolves the checkpoint, just point
`--classify_config` at the new dated config and it uses the new `best.ckpt`.

```bash
export AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret>
export RSLP_PREFIX=/weka/dfive-default/rslearn-eai PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
python -m rslp.landsat_vessels.scripts.run_recent_inference \
  --scene_csv /weka/dfive-default/yawenz/landsat/20260911_results/scene.csv \
  --out_dir   /weka/dfive-default/yawenz/landsat/<date>_results/new_classifier_<date> \
  --classify_config data/landsat_vessels/config_classifier_<date>.yaml \
  --det_thr 0.7 --cls_thr 0.99 --workers 3
```

Then diff `detections.csv` (kept counts) and eyeball `figures/<scene>/candidates_grid.png`
new-vs-baseline. Expect FP-heavy scenes to drop sharply and non-feedback scenes to stay
stable. `run_recent_inference` writes the exact `patched_classifier.yaml` it used — check
its `run_name` to confirm the right checkpoint was loaded. (~950 s/scene with 3 shards
sharing one GPU; that's just GPU contention.)

### 5. Publish the new version (e.g. `v1.1.1`)

`publish.py` plans by default (prints every action, touches nothing); add flags to apply.

```bash
python -m rslp.landsat_vessels.feedback.publish \
  --run-name olmoearth_base_layerdecay_<date> \
  --config   data/landsat_vessels/config_classifier_<date>.yaml \
  --old-run-name <run currently in the Dockerfile>   # e.g. olmoearth_base_layerdecay_20260908d
  --upload --patch            # then --build --push (or run the printed docker commands)
```

What it does:
1. **upload** `best.ckpt` →
   `gs://ai2-rslearn-projects-data/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_<date>/best.ckpt`
   (date is in the run name → date is in the GCS path).
2. **patch** `Dockerfile` (classifier `wget` `<project>/<old_run>` → `<project>/<new_run>`)
   and `config.py` (`CLASSIFY_MODEL_CONFIG` filename → the new dated config).
3. **build & push** via `docker compose -f rslp/landsat_vessels/docker-compose.yaml build`
   then `docker push landsat_vessels:latest` (needs a docker daemon + registry creds;
   usually run by hand from the printed commands).

Then **smoke-test the container endpoint** (port 5555) — not just the local checkout —
before redeploying to Skylight integration.

#### Where the version label lives

The `v1.x.x` string (what shows up as `rslearn_model_version` in the detection JSONs) is
**not** in this repo — the FastAPI app only declares an API `version="0.0.1"`. The model
version is assigned at **deploy time on the Skylight/sat-service side**, plus the **docker
image tag**. So to cut `v1.1.1`:
- tag the pushed image accordingly (pass `--image landsat_vessels:v1.1.1` to `publish.py`,
  or retag), and
- register/deploy that image to Skylight with model version `landsat_vessels_v1.1.1`
  (Skylight-side config — coordinate with the Skylight deploy owner).

The repo change per version is only *which checkpoint the image serves* (Dockerfile +
`config.py`). Version numbers are free-form deploy labels: a checkpoint-only refresh might
be `v1.0.1`; a larger change `v1.1.0`/`v1.1.1`. Use whatever the user specifies.

### 6. Close the cycle — bump the constants for next time

So the *next* loop pulls/patches against the newly-deployed model, update
`rslp/landsat_vessels/feedback/config.py`:
- `DEPLOYED_MODEL_VERSION` → the new version string (e.g. `landsat_vessels_v1.1.1`)
- `DEPLOYED_RUN_NAME` → `olmoearth_base_layerdecay_<date>`
- `BASE_CLASSIFIER_CONFIG` → `data/landsat_vessels/config_classifier_<date>.yaml`

and update the **Current state** section at the top of this file.

---

## Checklist for the next agent

- [ ] Confirm which model version is *currently deployed* (update §Current state).
- [ ] `pull` new feedback: `--username`, `--since <deploy date> --date-field submission`,
      `--keep bad_only`, `--model-version <deployed>`.
- [ ] `create_windows --materialize`; sanity-check window count + a `candidates` grid via
      `python -m rslp.landsat_vessels.feedback.visualize`.
- [ ] `add_to_training`; verify the config diff is *only* the new group + run_name.
- [ ] Train on a healthy GPU node; watch `val_correct_f1` + `best.ckpt`.
- [ ] Validate with `run_recent_inference` (new vs baseline) — FP scenes down, recall stable.
- [ ] Get user sign-off, then `publish` (upload + patch + build + push) with the right
      image tag; smoke-test the container.
- [ ] Coordinate the Skylight redeploy + version label.
- [ ] Bump the §6 constants + §Current state.

## Key paths

- Feedback tools: `rslp/landsat_vessels/feedback/` (`pull`, `create_windows`,
  `add_to_training`, `publish`, `visualize`; constants in `config.py`).
- Classifier dataset: `${RSLP_PREFIX}/datasets/landsat_vessel_detection/classifier/dataset_20250624`
  (windows under `windows/feedback_<date>/`).
- Checkpoints: `${RSLP_PREFIX}/projects/landsat_vessel_classification_v2/<run>/best.ckpt`.
- Served image: built from `rslp/landsat_vessels/Dockerfile` +
  `docker-compose.yaml` (image `landsat_vessels:latest`); checkpoints are `wget`ed from GCS at build.
- QA scenes: `/weka/dfive-default/yawenz/landsat/20260911_results/scene.csv` (6 scenes),
  baseline results alongside it.

## Known gotchas

- `rslp/landsat_vessels/__init__.py` is lazily imported (PEP 562 `__getattr__`) so the
  feedback tools don't drag in the heavy torch `predict_pipeline`. Keep it lazy.
  (This once broke `run_recent_inference.py`, which used
  `sys.modules["rslp.landsat_vessels.predict_pipeline"]`; it now imports the module
  explicitly — if you add scripts, import the module, don't read `sys.modules`.)
- The Skylight integration API returns `403` from compute (LB IP block) and production
  returns `401`; always enrich coordinates from GCS (`--source gcs`).
- Skylight's own `image_chips` PNGs are transient (cleaned up); render crops from the
  materialized windows (`feedback.visualize`) instead of relying on them.
- `get_cached_checkpoint` returns weka (local-fs) paths directly with no caching, so
  there is no stale-checkpoint risk when re-running inference.
