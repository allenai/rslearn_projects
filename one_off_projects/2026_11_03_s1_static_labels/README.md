Sentinel-1 vessel detection: same-track historicals and label cleanup
====================================================================

November 2026. This produced dataset `sentinel1_vessels/dataset_v1/20261103` and the
model in `data/sentinel1_vessels/config.yaml`.


Background
----------

After the production Sentinel-1 vessel model moved to one historical image and a 0.7
score threshold, users saw more false positives on wind turbines, platforms and islands.
On 34 scenes over offshore wind farms (Dogger Bank, German Bight, Stavanger), the model
made about 900 detections at 0.7 after the marine-infrastructure filter. About half of them
repeated at the same spot in three or more scenes.

The historical image is there so the model can tell vessels, which move, from fixed
objects. But the model used it only weakly. An object clearly visible in the historical
still scored at least 0.7 about half the time. We found two reasons:

- Static objects were labeled as vessels. About 5.7% of training labels sat on an object
  that looks the same in the historical image: platforms, turbines, islands, beacons and
  permanently anchored ships. Near mapped infrastructure, about 1 in 6 bright static objects
  was labeled "vessel".
- Historical images were chosen by orbit direction only, so some came from a different
  relative orbit (track), with different viewing geometry. Among the labels we reviewed,
  this was true of 15% of the first and 23% of the second training historicals.

Geolocation error between target and historical images was ruled out: the global offset is
0-2 px.


Dataset 20261103: same-track historicals
----------------------------------------

`dataset_v1/20261103` has the same 39,931 windows as `20250521`. Labels, masks and target
images started as hardlinks to 20250521, and every later label edit replaces the file, so
20250521 is unchanged.

`sentinel1_historical` holds up to four item groups, all from the target's relative orbit.
Time before the window start is split into 30-day periods (5-35, 35-65, ..., 125-155 days).
The periods are visited most recent first, and each period with a usable pass contributes
one group. A pass is a set of GRD frames acquired minutes apart. It is usable if its frames
cover at least 98% of the part of the window inside the target footprint. Within a period,
one usable pass is picked at random, seeded by the window name. rslearn's own prepare step
can't express this, so the rule lives in a custom prepare workflow:

    python -m rslp.main sentinel1_vessels prepare --ds_path /path/to/dataset

The 2026-11 build used an earlier version of that code. It searched the OEDatasets RTC
catalog and skipped 14 scenes missing from the AWS bucket that the pixels came from. On a
sample of 1,392 windows:

- The workflow reproduces the stored selection exactly in 53% of windows.
- Most other differences are reprocessed copies of the same acquisition: same pixels,
  different product name.
- In 7% of windows, one group picks a different pass from the same period.

Result: 98.6% of windows have four historicals, and 0.5% (184) have none. Group 0 comes from
the 5-35 day period in 98.7% of windows.

Training uses one historical at a time, chosen at random among the four for every example
(`data/sentinel1_vessels/20261103_static_labels/config_1hist_randhist.yaml`). This was the
largest single improvement: see that folder's README. Prediction uses one historical. The
prediction pipeline picks the best-overlapping scene from 30-60 days before the target. On
205 recent scenes that was always on the target's own track, even though the pipeline
doesn't check this.


Label cleanup history
---------------------

Labels went from 61,655 (20250521) to 57,611 after rounds 1 and 2, then 52,577 after
round 3. A dropped label's window stays in the dataset, so the object becomes a hard
negative. The root of `dataset_v1/20261103` has one GeoJSON point file per stage
(`labels_orig`, `labels_20261105`, `labels_20261107`). The final file records why each
label was kept.

Throughout, favyen reviewed samples in the review app (`review_app/`). His labels override
any automatic decision: infra, island and derelict drop the label; vessel and moored keep it.

Round 1 (2026-11-03): static labels
- Candidates: labels where the same object is visible in the old training historical (VV
  contrast at least 10 dB and 15x15 px target/historical NCC at least 0.7). That was 3,133
  train and 348 val labels.
- Within 100 m of a mapped marine infrastructure point: dropped (984).
- The rest: Gemini (gemini-4-argon) classified each label as fixed structure / island /
  permanently anchored vessel / vessel / unsure. It looked at chips of the target and the
  four same-track historicals. The first three classes were dropped.

Round 2 (2026-11-05): a second static test, and anchored vessels revisited
- A test on VH (and VV) against the same-track historical flagged 2,248 more labels that
  round 1 had not considered.
- The 336 round-1 "permanently anchored vessel" drops were reconsidered, because many were
  ordinary ships at anchor.
- Both sets went through a Gemini prompt with four answers (infrastructure / island /
  vessel / unsure); infrastructure and island were dropped. 1,322 labels were dropped and
  161 anchored ones restored.
- After rounds 1 and 2, 4,044 labels had been dropped: 984 near infrastructure, 167 by
  favyen's labels and 2,893 by Gemini.
- Adding Sentinel-2 imagery to the Gemini prompt did not help. Sentinel-2 imagery was
  aligned to the RTC-corrected label positions, but Gemini still separated infrastructure
  from berthed or anchored vessels no better than with Sentinel-1 alone.

Round 3 (2026-11-07): Gemini check of every label (`gemini_label_check.py`)
- Every label, kept or dropped (about 61K), went to one prompt with three answers: wrong /
  low_confidence / correct. The full prompt is in the script.
- "Correct" is strict for stationary ships. A moored ship counts only if it is absent from
  most historicals. An anchored ship counts only if most historicals don't show what looks
  like the same ship.
- Chips: 2.56 km, VV and VH, target plus up to four same-track historicals, with a red ring
  at the label. Media resolution was MEDIUM and no few-shot examples were used. This setup
  was chosen after comparing prompt variants on favyen's labeled items.
- Checks before the full run:
  - On 100 favyen-labeled false positives from wind-farm scenes, Gemini agreed 96 times.
  - In a blind check of 100 kept labels, all 50 that Gemini called correct were vessels.
    Of the 25 it called wrong, 11 were infrastructure or derelict, 3 moored and 7 vessels.
- Rules (`apply_labels.py`):
  1. favyen's label wins.
  2. A label dropped in rounds 1-2 comes back only if Gemini says correct.
  3. A kept label is dropped only if Gemini says wrong. Low confidence is kept; many are
     real vessels at the image border.
- Answers for the kept labels: 49,624 correct, 2,615 low confidence, 5,099 wrong.
- Gemini called 11 of the 4,041 dropped positions correct, and favyen reviewed them. He
  labeled 8 as vessel or moored, so they came back. 1 stayed dropped. The 2 he left
  unlabeled came back on Gemini's answer.
- Labels went from 57,611 to 52,577 (train 47,312, val 5,265). The rules apply to val as
  well, and the final model was evaluated on this val.
- Cost: about $810 at Vertex batch list price ($2 per M input tokens, $10 per M output,
  for about 370M input tokens). A promotion through 2026-11-30 credits back half.

Every original label, with why it was dropped, is in the Studio project "Sentinel-1 Vessel
Detection 20261103" (Skylight organization). There is one task per window, and each point's
`status` is `kept`, `favyen_label`, `near_infra` or `gemini`.


Results
-------

The final model is the SatlasPretrain model with a random same-track historical, trained on
the round-3 labels. It was compared with the previous production model on the round-3 val
set, using the most recent same-track historical.

- At 0.8 both models have about 0.89 recall; precision is 0.864 vs 0.812.
- Val mAP is 0.538 vs 0.527.
- On the 34 wind-farm scenes, it makes 265 detections at 0.8, none persistent. The old model
  made 916 at 0.7, 474 of them persistent.
- The five false positives users reported all score below 0.3.
- It detects 24 of 25 AIS-matched vessels; the old model detects 25. The miss is a vessel
  off SW Australia that this model scores 0.62.

The prediction pipeline now defaults to 0.8. See
`data/sentinel1_vessels/20261103_static_labels/README.md` for the per-threshold tables and
the comparison of training variants.


Code
----

- `rslp/sentinel1_vessels/prepare.py`: same-track historical selection (see above).
- `gemini_label_check.py`: renders chips, then builds, submits and collects Vertex AI batch
  requests for the round-3 label check. Set `GOOGLE_CLOUD_PROJECT` (Vertex) and, if needed,
  `GCS_SERVICE_KEY_FILE` (gsutil).
- `apply_labels.py`: the round-3 rules, plus `verify` to check the window label files
  against the root GeoJSONs.
- `review_app/`: the review app.
  - `build_chips.py` turns a list of (window, row, col) positions into chips and items.
  - `app.py` serves them (`python app.py --data_dir DIR`). Reviewers press keys to label
    each item as infra, island, derelict, moored or vessel.
  - The app can flip between the target and historical images, shows each image's relative
    orbit, and shows an "also in this historical?" test.

The round 1 and 2 filters (marine-infrastructure distance, the static-object tests, and the
earlier Gemini prompts) are not included. Their decisions are recorded in the label files
and the Studio project.
