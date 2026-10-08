# Landsat vessel classifier annotations

An app for labelling detector candidates as real vessels or false positives, and the
record of each annotation round. Each round writes a new group (e.g. `round1_20260803`)
into the classifier dataset
(`/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624`),
next to the earlier groups. Skylight feedback goes through `../feedback/` instead.

## Running the app

1. Create one 512 px @ 15 m window per detection in the pool, with the scene pinned and
   frozen scene-level splits (needs AWS credentials for the `usgs-landsat` bucket):
   `python -m rslp.landsat_vessels.scripts.create_round1_windows`
2. Render the views the app serves:
   `python -m rslp.landsat_vessels.scripts.render_annotation_assets`
3. Serve the app and open <http://localhost:8501>:

   ```bash
   export ANNOTATION_ROUND_DIR=/weka/dfive-default/yawenz/landsat/annotation_round1
   python -m uvicorn rslp.landsat_vessels.annotations.app:app --host 127.0.0.1 --port 8501
   ```

Keys `1 2 3 0` label the focused card (correct / incorrect / unsure / skip); `?` lists
all shortcuts. Each label is appended to a JSONL log and written into the window's `label`
layer (`writeback.py`). The log is the source of truth: `POST /api/resync` re-applies it
if a dataset write failed.

Training configs center-crop these 512 px windows with rslearn's
`Pad(size=N, mode="center")`, so they match the 64 px groups in a batch.

## Round 1 (2026-07-30 to 2026-08)

**Why.** The v1.2 classifier failed on hard negatives (val AP 0.96, but 0.85 on production
feedback), with dense false positives over Arctic melt ice and cloud/glint in the tropics.
The training data was small and skewed (2,462 windows, `selected_copy` 94% positive).

**Scenes.** 300 scenes over the Skylight marine-regions ROI, at most one per WRS-2
path/row: 200 T1/T2 (2024-01 to 2026-05) and 100 RT (2026-07). Stratified toward
false-positive sources: coastal 20%, storm 20%, ice 20%, glint 10%, cloud 15%, open 15%.

**Pool.** 17,460 detector candidates (5,692 passed the classifier). 2,000 were selected,
prioritizing classifier passes in implausible places, 0.40-0.85 boundary cases, a sample
of confident passes, and stratified easy negatives.

**Windows.** Group `round1_20260803`, 2,000 windows. RT products deleted since sampling
use the T1/T2 product of the same acquisition (`substituted_product`). Splits: train 1,306
/ val 266 / test 428 detections over 114 / 21 / 60 scenes.

**Result.** Worth ~0.05 val F1; part of the v1.0.0 classifier
(`data/landsat_vessels/config_classifier_20260908.yaml`).

**Next round.** The sample is ~60% polar because WRS-2 path/rows crowd toward the poles;
weight candidates by cos(latitude).
