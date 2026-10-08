# Landsat vessel classifier annotations

An annotation app for labelling detector candidates as real vessels or false positives,
and the record of each annotation round.

The classifier dataset
(`/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624`)
already held labels from before this app existed: the original recheck groups
(`selected_copy`, `phase2a_completed`, `phase3a_selected`) and the round-0 Skylight
feedback group `feedback_20260325`. We keep training on those groups. New annotations
from round 1 onward are made with this app and written as new groups next to them (e.g.
`round1_20260803`). Production false positives reported in Skylight go through
`rslp/landsat_vessels/feedback/` instead, which writes dated `feedback_<date>` groups into
the same dataset.

## Running the app

1. Create one 512 px @ 15 m window per detection to label, with the scene pinned:
   `python -m rslp.landsat_vessels.scripts.create_round1_windows` (needs AWS credentials
   for the `usgs-landsat` bucket). It reads the pool GeoJSON at `POOL_PATH` and also
   assigns frozen scene-level splits.
2. Render the views the app serves:
   `python -m rslp.landsat_vessels.scripts.render_annotation_assets`.
3. Serve the app and open <http://localhost:8501>:

   ```bash
   export ANNOTATION_ROUND_DIR=/weka/dfive-default/yawenz/landsat/annotation_round1
   python -m uvicorn rslp.landsat_vessels.annotations.app:app --host 127.0.0.1 --port 8501
   ```

The page shows 20 detections at a time: a pan-sharpened 1.92 km crop, the 7.68 km context
around it, a reflectance sparkline and map links. `1 2 3 0` label the focused card
(correct / incorrect / unsure / skip) and `?` lists all the shortcuts.

Each label is appended to a JSONL log and fsynced, then written straight into the window's
rslearn `label` layer (`writeback.py`), so the dataset is current as you annotate. The log
is the source of truth: if a dataset write fails, `POST /api/resync` re-applies the whole
log. Clearing a label, or moving a window to skip, removes its layer.

The windows are larger than the classifier's training input so annotators get context.
Training configs center-crop them with rslearn's `Pad(size=N, mode="center")`, which also
leaves them the same size as the 64 px groups (OlmoEarth needs every sample in a batch to
have the same height and width).

## Round 1: negatives-focused re-annotation (2026-07-30 to 2026-08)

**Why.** The OlmoEarth v1.2 classifier failed on hard negatives, not positives: val AP was
0.96 but AP on production feedback cases was 0.85, with a third of the negatives at
mid-to-high probability. Skylight showed dense false positives over Arctic melt ice and
cloud/glint false positives in the tropics. The training data was small and skewed
(2,462 windows; `selected_copy` 94% positive), and window/patch-size sweeps were all
within noise, so the bottleneck was data.

**Scenes.** 300 scenes over the Skylight marine-regions ROI: 200 T1/T2 (2024-01 to
2026-05) and 100 RT (2026-07; RT products are deleted once reprocessed, so only recent
ones exist). Stratified toward false-positive sources: coastal 20%, storm (40-60° lat)
20%, ice (≥55° lat) 20%, glint (≤25° lat) 10%, cloud (≥40% cover) 15%, open 15%. At most
one scene per WRS-2 path/row, so a scene-level split cannot leak a location.

**Candidates.** The pipeline ran on every scene keeping classifier-rejected candidates and
their probabilities: 17,460 detector candidates, 5,692 passed the classifier. v1.2
rejected polar ice decisively (5-7% pass rate) but passed 30-60% in warm-water storm,
glint and open scenes, a mix of real traffic and whitecap/glint false positives.

**Pool.** After capping each scene at 75 candidates, 2,000 detections were selected for
labelling (at most 500 RT), prioritized as: classifier passes in implausible places (on
land, ice, Southern Ocean, high Arctic, clutter scenes), all 0.40-0.85 boundary cases, a
capped sample of confident passes in trafficked water, and stratified easy negatives.

**Windows and splits.** Group `round1_20260803`, 2,000 windows, all materialized. RT
products deleted since sampling fall back to the T1/T2 product of the same acquisition
(recorded per window as `substituted_product`). Frozen scene-level splits, stratified by
stratum × tier: train 1,306 / val 266 / test 428 detections over 114 / 21 / 60 scenes.

**Result.** Adding `round1_20260803` to training is worth ~0.05 val F1 and is part of the
v1.0.0 classifier (`data/landsat_vessels/config_classifier_20260908.yaml`); see
`rslp/landsat_vessels/README.md`.

**For a later round.** The sample is ~60% polar because WRS-2 path/rows crowd toward the
poles; weight candidates by cos(latitude) for area-uniform sampling.
