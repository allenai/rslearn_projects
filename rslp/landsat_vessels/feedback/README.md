# Landsat vessel feedback loop

Once a model is deployed to Skylight, users flag false positives in the app. This loop
turns that feedback into training windows, retrains the classifier, and republishes the
Docker image. Repeat whenever enough new feedback has accumulated.

## Current state

- **Deployed:** `landsat_vessels_v1.0.0` (config `config_classifier_20260908.yaml`, run
  `olmoearth_base_layerdecay_20260908d`) on Skylight integration, at detector 0.7 /
  classifier 0.99.
- **Feedback groups:** `feedback_20260911`, `feedback_20260928`.
- **Latest retrain config (not yet published):** `config_classifier_20260928.yaml`, the
  v1.0.0 recipe plus both feedback groups.

## Setup

Run from the repo root with `RSLP_PREFIX=/weka/dfive-default/rslearn-eai`, AWS
credentials for the `usgs-landsat` bucket, and `gsutil` authenticated. Use a per-batch
working directory, e.g. `/weka/dfive-default/yawenz/landsat/feedback_<date>/`.

## Steps

**1. Pull feedback.** Export the in-app feedback CSV from the Skylight admin panel
(`/admin?selected-tab=in-app-feedback`), then filter it and add coordinates from the
sat-service detection JSONs on GCS:

```bash
python -m rslp.landsat_vessels.feedback.pull \
  --input <dir>/in_app_feedback.csv --out <dir>/feedback_<date>.csv \
  --environment integration --source gcs \
  --username yawenz@allenai.org --since <deploy date> --date-field submission \
  --keep bad_only --model-version landsat_vessels_v1.0.0
```

`--model-version` matters: integration can serve several model versions, and only the
deployed model's mistakes should be added.

**2. Create windows.** One 64 px @ 15 m window per row in group `feedback_<date>`, with a
`label` layer, then materialize the imagery:

```bash
python -m rslp.landsat_vessels.feedback.create_windows \
  --csv <dir>/feedback_<date>.csv --group feedback_<date> --materialize
```

**3. Generate the training config.** Writes `config_classifier_<date>.yaml` with the new
group added and `run_name: olmoearth_base_layerdecay_<date>`. Pass the previous cycle's
config as `--base-config` to keep its feedback groups (the default is the v1.0.0 config),
and delete the superseded config afterwards.

```bash
python -m rslp.landsat_vessels.feedback.add_to_training --group feedback_<date> \
  --base-config data/landsat_vessels/config_classifier_<previous date>.yaml
```

**4. Retrain (1 GPU).**

```bash
rslearn model fit --config data/landsat_vessels/config_classifier_<date>.yaml
```

The best checkpoint (by `val_correct_f1`) lands at
`$RSLP_PREFIX/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_<date>/best.ckpt`.

**5. Validate.** Temporarily point `CLASSIFY_MODEL_CONFIG` in `config.py` at the new
config and run the feedback scenes and the scenario-check scenes (listed in
`ai2_docs/landsat_vessels/train_eval.md`) through
`python -m rslp.main landsat_vessels predict --scene_id <id> --json_path <out.json>`.
Compare detection counts with the deployed config: feedback scenes should drop, the rest
should stay in their expected ranges. `feedback/visualize.py` renders the feedback windows
as a grid.

**6. Publish.** Prints the plan by default; the flags perform each step.

```bash
python -m rslp.landsat_vessels.feedback.publish \
  --run-name olmoearth_base_layerdecay_<date> \
  --config data/landsat_vessels/config_classifier_<date>.yaml \
  --upload --patch   # then --build --push
```

This uploads the checkpoint to GCS, points the Dockerfile and `config.py` at the new run,
and builds and pushes the image. Smoke-test the container endpoint before redeploying,
and coordinate the version label with the Skylight deploy owner.

**7. Close the cycle.** Update `DEPLOYED_MODEL_VERSION`, `DEPLOYED_RUN_NAME` and
`BASE_CLASSIFIER_CONFIG` in `feedback/config.py`, and the current state above.
