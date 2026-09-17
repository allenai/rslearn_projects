# Landsat vessel classifier — annotation rounds

Plans and records for re-annotating the vessel classifier dataset. Each round gets a
section here; scripts live in `rslp/landsat_vessels/scripts/`.

## Round 1 — negatives-focused re-annotation (started 2026-07-30)

### Why

The current classifier (OlmoEarth, `config_classifier_20260616.yaml`) fails on hard
negatives, not on positives:

- Val AP is 0.96 but test AP is 0.85. On the val split, negatives pile up near
  prob 0 and any threshold works; on the test split (production feedback cases),
  a third of the negatives sit at mid-to-high probability where no threshold can
  separate them.
- Production Skylight maps (July 2026) show the same thing at scale: dense
  uncorrelated detections over Arctic melt ice (Canadian archipelago, Hudson Bay,
  Greenland), plus cloud and glint false positives around tropical island EEZs.
- The training data is small and skewed: 2,462 windows, with `selected_copy` 94%
  positive and `phase3a_selected` 100% negative. The test split is 51 windows from
  a single group — too small to compare models on.

Window/patch-size sweeps (64/4 vs 32/2 vs 16/1) all land within noise of each
other, so the bottleneck is data, not architecture.

Positive coverage from `selected_copy` and `phase2a` is considered adequate. Round 1
targets **diverse, production-realistic negatives**.

### Strategy

1. **Sample T1/T2 scenes over the Skylight marine-regions ROI**
   (`scripts/sample_scenes.py`). T1/T2 instead of RT because RT products get deleted
   from AWS; the `usgs-landsat` bucket only holds T1/T2, so the sample is reproducible
   forever. Stratified toward known false-positive sources:

   | stratum | share | definition | targets |
   |---|---|---|---|
   | coastal | 20% | footprint touches land | clutter, islands, reefs |
   | storm   | 20% | 40–60° abs latitude | whitecaps |
   | ice     | 20% | ≥55° abs latitude, non-polar-darkness months | melt ice + ice edge |
   | glint   | 10% | ≤25° abs latitude | sun glint |
   | cloud   | 15% | anywhere, scene cloud ≥40% | cloud false positives |
   | open    | 15% | anywhere in ROI | unbiased background |

   No cloud filtering in any stratum except cloud's minimum: Skylight runs every
   scene that arrives, so the sample matches the production input distribution.
   Scene picked uniformly at random within a random month window (2024-01 to
   2026-05). One scene max per path/row, so a later scene-level split cannot leak
   a location across train/test. Ice months deliberately include melt season
   (e.g. July in the Arctic) — that is where production false positives peak.

2. **Run the detector on the sampled scenes** to generate candidate detections.
   The classifier's job is filtering detector output, so negatives must come from
   the detector's actual output distribution. Cap annotated detections per scene
   (~50–100) so a single stormy or icy scene cannot dominate the dataset.

3. **Annotate with AIS-assisted triage.**
   - AIS-correlated detections (Skylight): auto-positive, human-verify a sample.
   - Uncorrelated detections: human review. Uncorrelated is NOT auto-negative —
     dark fishing fleets (SE Asia, West Africa) are real vessels without AIS,
     and erasing them is the one failure the classifier must not have.
   - Sort the review queue by the current classifier's prob(correct): high-prob
     false positives are the most valuable hard negatives; near-0.5 cases are the
     informative ambiguous ones. Requires `prob_property: "prob"` in the classifier
     config so probabilities are written with predictions.

4. **Freeze a permanent test set before training on anything.** Split at the
   scene level (all detections from one scene stay in one split), stratified by
   region/condition, several hundred detections, natural class ratio. Keep the old
   `feedback_20260325` test set as a secondary RT-distribution check.

### Annotation manifest

Detections are handed to annotators as a manifest (CSV + GeoJSON,
`scripts/make_annotation_manifest.py`) with one row per detection carrying
everything needed to acquire the imagery independently: lat/lon, acquisition
timestamp, scene_id, WRS-2 path/row, the product's S3 prefix in the usgs-landsat
bucket, pixel col/row + CRS, detector score, classifier label and prob(correct),
stratum/region/cloud metadata, a crop image path, and empty
human_label/annotator/notes columns. Rows are ordered confident → ambiguous →
likely-negative so the highest-value reviews come first.

This requires the pipeline run with `include_rejected=True` (annotation mode):
the classifier's verdict and probability are recorded on every detector candidate,
and rejected candidates stay in the JSON output. Production behavior is unchanged
by default. Requires `prob_property: "prob"` in the classifier config (added
2026-07-31).

### Files

- `scripts/sample_scenes.py` — stratified scene sampler (writes scene CSV)
- `scripts/make_annotation_manifest.py` — detection JSONs + scene CSV -> annotation manifest
- `scripts/map_scene_sample.py` — world-map panels of strata pools + selected scenes
- `/weka/dfive-default/yawenz/landsat/World_Marine_Regions.geojson` — Skylight ROI
- `/weka/dfive-default/yawenz/landsat/scene_sample_v1.csv` — round 1 scene sample (200 scenes)
- `/weka/dfive-default/yawenz/landsat/strata_pools_map.png` — strata pool map
- `/weka/dfive-default/yawenz/landsat/scene_sampling_cache/` — WRS-2 shapefile + S3 listing cache

### Round 1 sample (scene_sample_v1.csv, 200 scenes, sampled 2026-07-31)

- stratum: coastal 40, storm 40, ice 40, cloud 30, open 30, glint 20 (all quotas filled)
- latitude: 119 polar (>=55), 43 mid, 38 tropics
- cloud cover: 49 clear (<20%), 52 partly (20-60%), 43 cloudy (60-90%), 56 overcast (90%+)
- 90 T1 / 110 T2; 88 Landsat-8 / 112 Landsat-9; 2024: 65, 2025: 101, 2026: 34;
  all 12 months represented
- top regions: Southern Ocean 16, NW Passages 14, Ross Sea 12, Weddell Sea 11,
  North Atlantic 9

### RT supplement (scene_sample_rt_v1.csv, 100 scenes, sampled 2026-07-31)

Production Skylight runs on Real-Time (RT) tier products, and the feedback datasets
contain RT imagery, so the annotation pool should cover RT alongside T1/T2. RT is a
transient tier: products processed with predicted ephemeris sit in usgs-landsat only
until the T1/T2 reprocess supersedes them (days to ~3 weeks; most scenes now go
direct-to-T1 same day), and superseded RT is deleted with no public archive. So a
2024-2026 RT sample is impossible — only the trailing ~2 weeks exist at any time.

`scripts/sample_rt_scenes.py` therefore inverts the T1/T2 sampler: it enumerates
every RT product currently live over the ROI (2,583 on 2026-07-31), then fills the
same strata quotas from that pool, excluding path/rows already used by
scene_sample_v1.csv (split hygiene). Result: 100 scenes, all 2026-07, quotas filled
(coastal 20, storm 20, ice 20, cloud 15, open 15, glint 10); 41 polar / 30 mid /
29 tropics. Re-run the script in later weeks to top up as new RT appears.

Because RT products expire, the detector must run promptly after sampling, and by
annotation time the manifest's s3_prefix for RT rows may be gone — annotators can
substitute the T1/T2 reprocess of the same acquisition (same pixels to within
geolocation refinement) or use the saved crops.

### Round 1 execution notes (2026-07-31, run in progress)

**Detection run.** All 300 scenes (200 T1/T2 + 100 RT) run locally on one H100 via
`scripts/run_round1_detections.py`, sharded round-robin across 10 parallel workers
(6 T1/T2 + 4 RT; ~1.1 GB GPU each, worst case ~40 GB — network-bound, not
GPU-bound). Outputs land in
`/weka/dfive-default/yawenz/landsat/round1_detections/{json,crops}`; scratch is
local disk, deleted per scene, so full scenes are never stored. RT ran first since
those products expire.

**Interim aggregate (~70 scenes in):** classifier prob buckets per detector
candidate, by stratum/tier:

| stratum/tier | scenes | cands | pass (>=.85) | ambig (.4-.85) | reject (<.4) |
|---|---|---|---|---|---|
| coastal T1/T2 | 40 | 2,650 | 245 (9%) | 180 | 2,225 |
| coastal RT | 17 | 2,635 | 1,105 (42%) | 378 | 1,152 |
| storm T1/T2 | 14 | 470 | 320 (68%) | 49 | 101 |

- **Arctic/ice: v1.2 is strong.** Melt-ice flood scenes produce hundreds-to-1,109
  detector candidates and 0-3 classifier passes each. The handful that do pass are
  premium hard negatives.
- **The v1.2 leak is warm-water**: every scene passing 30+ detections is |lat|<55
  (Bahamas RT 636/848, W-Med RT 328/338, Black Sea 92/94, Aegean 166/527). Crop
  inspection: N-Atlantic and Black Sea storm passes are mostly *real vessels with
  wakes* (trafficked seas); far-S-Atlantic (49S) passes at prob 0.97-1.00 are
  *featureless rough water* — whitecap/glint FPs, the top annotation targets.
  AIS correlation at triage separates the two cheaply.
- RT coastal pass-rate (42%) >> T1/T2 coastal (9%) mostly because the RT sample is
  all July (summer/glint season, more small-boat traffic), not tier physics.

**Manifest capping fixed** (`make_annotation_manifest.py`): the per-scene cap is
now bucket-stratified — keep all confident_vessel, then ambiguous, fill the rest
with sampled likely_negatives — so flood scenes can't dilute hard FPs out of the
manifest. Also fixed `crop_fname` to read the Landsat pipeline's `crop_fnames.rgb`.

**Production deployment gap (major finding).** Skylight's July 2026 Arctic FP
flood is NOT current-pipeline behavior: the served Docker image still runs the
2024-09-19 classifier. `rslp/landsat_vessels/Dockerfile` downloads
`rslearn-landsat-recheck/phase123_20240919_01_copy/best.ckpt` for the classifier;
the v1.1/v1.2 swaps (commits 852dffa5, 7349d982) landed 2026-07-16, one day after
the Dockerfile was last touched. A rebuild from HEAD would crash (config restores
`landsat_vessel_classification_v2/olmoearth_base_v1.2_final_20260616/best.ckpt`,
which the Dockerfile never fetches). Fix checklist:
1. upload that checkpoint (on weka under rslearn-eai/projects/...) to
   `gs://ai2-rslearn-projects-data/projects/landsat_vessel_classification_v2/olmoearth_base_v1.2_final_20260616/best.ckpt`
2. point the Dockerfile's classifier wget at it (and verify
   `OlmoEarth(model_id=OLMOEARTH_V1_2_BASE)` resolves offline in-container)
3. rebuild and smoke-test the **container endpoint** (`evaluation/smoke_test.py`
   only exercises the local checkout — that's how this drift went unnoticed);
   make endpoint smoke-testing a standing pre-deploy step
4. redeploy; expect the high-latitude uncorrelated flood to collapse

Deploying v1.2 is independent of (and faster than) this annotation round; the
round's marginal value is the residual warm-water leak above.

**Smoke test** (local pipeline, v1.2, interim): ice scene PASS (24 det -> 0 cls),
whitecap scene 001090 PASS (296 det -> 1 cls). So v1.2 handles those regimes;
prior smoke failures were likely observed against the stale deployed image (or an
older local config) — re-check which environment a failing run used.

**Smoke test, final (local pipeline = v1.2, 2026-07-31): 4 PASS / 3 FAIL.**

| scene | scenario | expected | classifier | result |
|---|---|---|---|---|
| LC09_L1GT_129107_20241104 | mostly ice | [0,10] | 0 | PASS |
| LC09_L1TP_001090_20241103 | mostly whitecaps | [0,10] | 1 | PASS |
| LC09_L1TP_193021_20241104 | some vessels | [20,50] | 49 | PASS |
| LC09_L1TP_170084_20241103 | some vessels | [20,50] | 42 | PASS |
| LC09_L1TP_177081_20241104 | mostly whitecaps | [0,10] | 20 | FAIL |
| LC09_L1TP_010012_20241102 | islands + ice | [0,10] | 22 | FAIL |
| LC09_L1TP_193030_20241104 | some vessels | [20,50] | 127 | FAIL |

Reading: melt-ice flood filtering passes; the failures are the residual v1.2
modes (selective whitecap leak, island/ice-edge clutter, over-passing in busy
warm coastal water) — the same modes the round-1 aggregates show, i.e. what this
annotation round targets. "Filters Arctic ice" and "fails smoke" are consistent.

**Distance-to-coast (Skylight public API, scripts/coast_distance.py).** Skylight
production keeps a detection only if over water AND >= 10 m from coastline. Cross-
tab over 8,805 round-1 detections (135 scenes): 311 on land (7 classifier-passed),
22 under 10 m, 160 at 10-100 m (43 passed — docked ships/piers/rocks survive
production's rule), 58% of all passes are > 2 km offshore. So on-land clutter is a
small slice of the production FP problem; the classifier carries the rest. The
manifest is enriched with `distance_to_coast_m` / `land_cover_class`, and
select_annotation_pool treats on-land classifier passes as automatic
suspect_confident (certain hard negatives).

### Round 1 final numbers (run completed 2026-07-31)

All 300 scenes (200 T1/T2 + 100 RT) processed, **0 failures**. 17,460 detector
candidates, 5,692 passed the classifier (32.6%). By stratum/tier
(scenes / candidates / passed):

| stratum | T1/T2 | RT |
|---|---|---|
| coastal | 40 / 2,650 / 245 | 20 / 2,974 / 1,359 |
| storm   | 40 / 1,217 / 756 | 20 / 1,835 / 1,138 |
| ice     | 40 / 1,110 / 97  | 20 / 1,370 / 87 |
| glint   | 20 / 556 / 305   | 10 / 1,029 / 284 |
| cloud   | 30 / 1,225 / 118 | 15 / 410 / 42 |
| open    | 30 / 1,607 / 422 | 15 / 1,477 / 839 |

Manifest: 7,906 rows after the bucket-stratified 75/scene cap (3,676 confident /
749 ambiguous / 3,481 likely-negative; 9,554 dropped over cap, nearly all easy
negatives from flood scenes). Coast enrichment: 582 rows not over water, 36 over
water but < 10 m from coast.

**Annotation pool** (`select_annotation_pool.py`, budget 2,500):
`/weka/dfive-default/yawenz/landsat/round1_annotation_pool.{geojson,csv}` —
2,495 detections, ordered as the review queue (slice priority, then descending
classifier prob):

- suspect_confident 279 — classifier-passed in implausible contexts (on land per
  Skylight land-cover, ice stratum, Southern Ocean <= -45, high Arctic >= 66, or
  clutter-field scenes with <10% pass rate among >= 30 candidates). Gold hard
  negatives (note: high-Arctic slice includes some real Barents-fleet vessels).
- ambiguous 749 — all prob .40-.85 decision-boundary cases.
- plausible_confident 504 — passed detections in trafficked water, capped 5/scene
  (positives already well covered; this is an FP-contamination audit +
  dark-vessel spot check).
- easy_negative 963 — rejected candidates, stratified by stratum x tier, half
  near-boundary / half random.

Balanced across all 12 stratum x tier cells (137-297 each). Expected label mix
roughly 75-80% negative. No AIS triage was available (these scenes never went
through Skylight correlation), so slices set human review priority instead.

Selection-review galleries (criteria, world map, per-slice crops with
location/timestamp/prob): https://claude.ai/code/artifact/73b17fbb-ae54-4494-ab83-04299d5a9c4e

**Pool v2** (`round1_annotation_pool_v2.{geojson,csv}`): same selection with
`--budget 2000 --max_rt_rows 500`. The v1 pool is 46% July-2026 rows (the RT
supplement is inherently single-month, and RT scenes over-produced passes), so
v2 caps RT at 500 to bound the temporal/tier skew: 2,000 rows, exactly 500 RT,
suspect_confident (279) and ambiguous (749) kept in full — the cap trims the RT
audit slice and shifts negative fill to T1/T2. v2 galleries:
https://claude.ai/code/artifact/1dc1e4d7-109a-4258-8abf-e09f86e7984d

Headline model observation from the full run: v1.2 rejects polar ice decisively
(ice stratum pass rate 5-7%), while warm-water storm/glint/open scenes pass
30-60%+ — a mix of real traffic and whitecap/glint FPs; the pool concentrates
human effort on exactly that boundary.

### Open questions

- The sample is ~60% polar. Two causes: polar coasts split into many path/rows
  (coastal pool), and WRS-2 paths converge toward the poles, so uniform sampling
  over path/rows oversamples high latitudes in every stratum (visible in the
  "open" pool map). Acceptable for round 1 given the Arctic FP flood; if a later
  round needs area-uniform sampling, weight candidates by cos(latitude).
- Optional: weight future sampling by Skylight event exports (scenes with high
  uncorrelated-to-correlated ratios in implausible places), and re-materialize past
  production false positives from the T1/T2 versions of the same acquisitions.

### Windows and annotation platform (built 2026-08-03)

The 2,000 pool detections are now rslearn windows in group **`round1_20260803`** of
`/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624`,
with imagery acquired and the annotation app serving them.

**Window geometry: 512 px @ 15 m (7.68 km), one window per detection.** The classifier
trains on 64 px, but annotation needs context — is this a lone hull, or one bright pixel
in a melt-ice field? A single 512 px window serves every view, so nothing has to be
re-acquired to change the training window size:

| view | extent | where it comes from |
|---|---|---|
| crop | centre 128 px = 1.92 km | the annotation card and drawer |
| zoom-out | full 512 px = 7.68 km | the context card and drawer |
| training input | centre 64 px = 960 m | `CenterCrop` at read time |

Windows are on the same grid as `feedback_20260325` (15 m, UTM/UPS per detection from
`get_utm_ups_crs`), so the `landsat` layer materialises as B8 at 15 m (512²) plus the
other ten bands at 30 m (256²) — 1.75 MB per window, **3.5 GB** for the group.

**Items are pinned, not matched.** Each pool row names the scene its detection came
from, so `create_round1_windows.py` writes `items.json` directly from the data source's
`get_item_by_name` instead of running `rslearn dataset prepare`, which would re-match by
time and space and could pick an overlapping neighbour scene. 57 of the 195 scenes are RT
products USGS has since deleted; those fall back to the definitive T1/T2 product of the
same acquisition, recorded per window as `substituted_product` and shown in the app's
drawer. Materialize: **2,000/2,000, 0 failures** (~6.5 h wall clock, 32 workers).

**Frozen splits.** Assigned at scene level (all detections from one acquisition stay in
one split), stratified by stratum × tier, quota by detection count, deterministic from a
sha256 of the scene id. Result: **train 1,306 / val 266 / test 428** detections over
114 / 21 / 60 scenes. This is the permanent test set the round-1 plan called for — 428
detections over 60 distinct scenes, versus the old 51-window single-group test split.

**Annotation app.** FastAPI + one static page, served from this machine:

```bash
export ANNOTATION_ROUND_DIR=/weka/dfive-default/yawenz/landsat/annotation_round1
/weka/dfive-default/yawenz/.venv-rslearn/bin/python -m uvicorn \
  rslp.landsat_vessels.annotations.app:app --host 127.0.0.1 --port 8501
```

Then open <http://localhost:8501> (VS Code forwards the port automatically over the SSH
session). A gallery of 20 detections per page: pan-sharpened RGB crop, the 7.68 km
context beside it with the crop extent boxed, a reflectance sparkline, Google Maps and
satellite links for the location, and `✓ ✗ ? »` plus a reason menu on every card. Enter
opens a drawer with the full-size views, both spectral panels, the sampled band values
and all metadata. Keyboard: arrows move the focused card, `1 2 3 0` label it and advance,
`i c g w a k o` set the reason, `,` `.` page, `v` cycles crop view (auto stretch →
production stretch → B8), `[` `]` adjust gain, `?` lists everything.

**Labels reach the dataset as you annotate.** Each keystroke is appended to
`annotation_round1/labels/round1_20260803_labels.jsonl` and fsynced, and then written
straight through into that window's rslearn `label` layer — so no separate step is needed
to make an annotation real. The order is deliberate: the log is durable *before* the
dataset write is attempted, so a dataset problem degrades to "reconcile later" and never
to lost work. Failures are counted and surfaced (`/api/label_layers` reports what is
actually on disk versus what the log says; `POST /api/resync` re-applies everything).

Retractions propagate too: clearing a label removes its layer, and moving a labelled
window to `skip` removes it as well, so the dataset never keeps a judgement the annotator
has taken back.

**Two stretches, three views, both panels.** The card shows an auto stretch (each crop
scaled to its own 2–98% DN range); `v` switches to the fixed `(DN−5000)×255/12000`
production window and then to B8, and the crop and the 7.68 km context switch together
(hence `crop_*` and `zoom_*` renderings of each). The difference is not cosmetic: in the
Gulf of Bothnia example the vessel and its wake are obvious in auto and nearly invisible
in production, which is exactly the class of detection the model is judged on.

Every RGB view — auto and production, crop and zoom — is pan-sharpened with the same
Brovey transform the production pipeline uses, `band × B8 / mean(B2,B3,B4)` applied after
the 8-bit stretch (`predict_pipeline._write_detection_crop`). The production crop is
therefore the pipeline's own crop: diffed against the round-1 detection PNGs the earlier
artifact displayed, 12 of 12 match to **0.43/255 mean absolute difference** (max 1.04),
the residual being a ≤1 px grid offset because window centres here come from lat/lon
while the pipeline used `pixel_col`/`pixel_row`.

**Reconciling.** `apply_round1_labels.py` re-applies the whole log (last record per window
wins), retracts stale layers, reports the label mix by split/slice/reason and the count of
classifier-passed detections a human called incorrect, and writes
`round1_20260803_pool_labeled.{csv,geojson}` while leaving the original pool alone. It is
idempotent — after normal write-through use it reports everything already current. What a
label layer contains is decided once, in `annotations/writeback.py`, shared by both paths
so they cannot drift.

**Mixed window sizes when training — measured, not assumed**
(measured with a batch of 8 over `round1_20260803` + `feedback_20260325`):

| setup | collation | OlmoEarth forward |
|---|---|---|
| mixed 512 px + 64 px, no transform | OK — shapes `(11,64,64)` and `(11,512,512)` in one batch | **RuntimeError**: expanded size 512 must match existing size 64 |
| mixed, with `CenterCrop(64)` | OK — all `(11,64,64)` | OK |
| 512 px alone, with `CenterCrop(64)` | OK — all `(11,64,64)` | OK |

So rslearn's *loader* is size-agnostic: `train.data_module.collate_fn` transposes a batch
into lists of per-sample dicts and never stacks tensors, so mixed sizes pass through it
untouched. The constraint is in the *encoder*: `OlmoEarth` takes height/width from the
first sample of the batch, allocates a buffer of that shape per sample and stacks them, so
samples that disagree spatially fail there. `CenterCrop` removes the disagreement, which is
what makes one 512 px group trainable alongside the 64 px groups. Keep `crop_size: 64`
while any 64 px group is in the run; to train at 128 or 256, restrict `groups` to
`round1_20260803`. (`SplitConfig.crop_size` is not an alternative — it takes a *random*
sub-window, which on a 512 px window usually misses the vessel.)

`data/landsat_vessels/config_classifier_round1_20260803.yaml` is `config_classifier_20260616.yaml`
plus the new group and that transform, ready for round-2 training once labelling is done.

**Files**

| file | role |
|---|---|
| `scripts/create_round1_windows.py` | pool → 2,000 windows + pinned items + frozen splits + index JSON |
| `scripts/render_annotation_assets.py` | windows → 3 crop + 3 zoom renderings (342 MB) + `spectra.json` |
| `annotations/app.py` | the annotation server (session, batched page fetch, label append + write-through) |
| `annotations/static/index.html` | the gallery page, drawer and charts |
| `annotations/writeback.py` | what a label layer contains; shared by the app and the reconciler |
| `annotations/check_page.py` | static checks on the page (ids, brackets, CSS vars) — no node on this host |
| `scripts/apply_round1_labels.py` | reconcile the labels JSONL against the windows + labelled pool copies |
| `scripts/verify_round1_windows.py` | reads windows the way training will, with labels |
| `rslp/landsat_vessels/transforms.py` | `CenterCrop` |

Spectral curves are reused from the round-1 spectral pipeline's cache
(`landsat/spectral/cache/`, TOA reflectance from each scene's MTL coefficients, thermal
as brightness temperature) — all 2,000 detections resolved, so no band was re-downloaded
for the app.

### Training-side fixes to pair with the new data (from the 2026-07-30 diagnosis)

- All runs peak at epoch ~20 (the unfreeze point) and overfit afterwards: cap
  max_epochs near 30, add EarlyStopping on val_loss, and either skip unfreezing the
  encoder or use unfreeze_lr_factor ≥ 100.
- `positive_class_threshold` does not affect reported metrics (torchmetrics use
  argmax); it only controls what `predict` writes — i.e. what the pipeline filters
  on. Pick the operating threshold from a PR sweep on val
  (`evaluation/pr_curve.py`), not by hand.
