Sentinel-1 static-label cleanup: detailed write-up
=================================================

This file was written by an agent (Claude) for favyen in November 2026, card t-cpb5wt on the
agent task board. `README.md` is the short version. The scripts, logs and intermediate
outputs are not in this repository. They are in the agent session folder:

    /weka/dfive-default/agent/favyen/2026-11-02-favyen-s1-vessel-false-positives/

Its `README.md` is the running log. Subfolders: `review/` (review app, Gemini, label edits),
`rot/` (rotation test), `newhist/` (the 20261103 build). Large work files are in
`/weka/dfive-default/favyen/s1fp_work/`. All paths below are relative to the session folder
unless absolute.


1. The report
-------------

Skylight users said the rslearn_projects Sentinel-1 vessel model (Skylight int) flags more
fixed infrastructure as vessels.

Model history: on 2026-06-30 (`099857e1`) the 2-historical model was replaced by the
1-historical model (`projects/20260626_num_historical/..._1hist`). On 2026-07-15
(`bde79307`) the threshold was lowered from 0.9 to 0.7.

favyen asked two questions:
- Is a 2-historical model better at a threshold with matched val recall?
- Do we reproduce the reported detections, and could the historical image be too old or
  misaligned?


2. Investigation (2026-11-02/03)
--------------------------------

**Scenes.** We used 34 target scenes (2026-10-19 to 11-02) over the Dogger Bank, German
Bight and Stavanger AOIs, and ran 11 model × historical-image variants on them (predict job
01M6F885QR5PF5WS4D7QX7EEDS; `analyze.py`, `plan_scenes.py`, `scene_run.py`).
- All 5 Skylight example events reproduce, within 4–8 m of Skylight's positions.
- Skylight int picks one historical: the most recent frame in [t-30d, t) with ≥90%
  footprint overlap (skylight-monorepo `sentinel1_processor.py`). The rslp scene-ID
  pipeline instead picks the best-overlap scene in [t-60d, t-30d] (same orbit direction).

**Alignment is not the problem.** Bright point targets have a median offset of 0 px between
target and historical (sometimes ±1–2) for every historical choice
(`bright_point_offsets.py`). In training windows the median offset is also 0
(`align_check_train.py`). The old vessel-detection-sentinels pipeline's dot-product shift
search finds 0–2 px and changes nothing.

**The historical is not stale.** In 64×64 crops (`crops64.py`, checked by eye), the
persistent high-score detections (Dogger turbines, an unmapped German Bight wind farm at
about 54.26–54.33N, 6.12–6.35E) show the same bright point at the same pixel in the
historical.

**2-hist vs 1-hist** (val jobs 01M6F93RWRR9JNRAR5SV0RS440, 01M6F93T5R77YQG1E7A23BZ5A1,
01M6F93WVE36AHMTRNFMSRFYE4, 01M6F93VMSHQDM1EBRKK54HKGC):
- val mAP is .503 / .508 / .503 / .502 (current 1hist, June 2hist, old 2hist, 0hist).
- At recall matched to current@0.7 (P .783, R .853), the number of persistent detections
  (≥3 scenes within 100 m) on the 34 scenes is: current 494, June 2hist 427, old 2hist 473,
  0hist 432.
- So a 2-historical model gives no real gain. The threshold matters much more: current
  model + recent historical gives about 460 persistent detections at 0.7 vs 43 at 0.9.

**Does the model use the historical?** (`hist_sensitivity.py`) We ran the same model and
target with four different historicals, on 3,055 objects.
- Object clearly present in the historical (≥15 dB): 52% score ≥0.7, 10% score ≥0.9.
- Object clearly absent (<6 dB): 72% score ≥0.7, 51% score ≥0.9.
- For Dogger turbines, the share scoring ≥0.7 is the same whether or not the turbine is in
  the historical.
- So the model uses the historical only weakly.

**Rotation test** (`rot/README_rot.md`). Each of (t, h1, h2), with all three ≥15 d apart,
was scored as the target with the other two as historicals (2hist model).
- Near mapped infrastructure, the model calls the object a vessel whichever date is the
  target: all three ≥0.5 for 70%.
- Pixels for replacement scenes came from AWS `sentinel-s1-l1c` windowed reads. These match
  the Copernicus-derived training rasters (corr ≥0.9998).


3. The training labels (root cause)
-----------------------------------

Dataset `dataset_v1/20250521`: 36,099 train windows, 55,000 train labels, all
`category: vessel` (the dataset has no other category). Scripts:
`train_label_static_check.py`, `train_static_ncc.py`, `train_static_summary.py`.

How the 3,133 static labels were found:
- **Brightness (no model involved).** In dB (20·log10 DN), contrast = max of the 5×5 box at
  the label minus the median of the 61×61 box. A label is "bright in the historical" if the
  VV contrast in `sentinel1_historical` is ≥10 dB: 4,532 labels.
- **Shape.** NCC between the 15×15 VV+VH patches of the target and the historical (each
  patch mean-subtracted), best over ±2 px shifts. NCC ≥0.7 means the same object is in the
  same place: 3,133 labels (5.7% of train labels).
- 876 of them are within 100 m of the 2026-10-12 marine infrastructure map (Satlas): the
  "infra" group. The other 2,257 are the "other" group.
- For comparison, 4,154 unlabeled static bright objects at mapped infrastructure are
  correctly left unlabeled. So about 17% of identical-looking static objects at mapped
  infrastructure were labeled vessel.
- The 20250521 historicals were chosen by orbit direction only. Among these labels, hist1
  is on a different relative orbit than the target for 479/3,133 and hist2 for 732.


4. Manual review and Gemini
---------------------------

**Review app** (`review/app.py`, `index.html`, `build_review_set.py`; a throwaway Flask app,
now stopped).
- Each group is shuffled (seed 0). The view shows target + training historicals as 96×96 and
  256×256 chips, VV/VH, with the track shown.
- favyen labeled 241 items (`review/labels.json`): infra 199, vessel 30, moored 6, derelict
  (= permanently anchored) 3, island 3.
- Infra group: 98/99 were infrastructure (one moored vessel). favyen decided to delete all
  of them.
- In the other group, many "infra" labels are not strictly infrastructure (jetties, coast,
  anchorages). They are still delete labels.
- favyen's priority: never delete correct open-water vessel labels. Deleting moored vessels
  is acceptable.

**Gemini round 1** (`review/gemini_eval.py`, `gemini_results.md`). 50 items, 6 variants,
target + 2 old historicals, about $5. `gemini-4-argon` was best: 94% on favyen's labels,
6/7 vessels kept. Confidence scores were uninformative.

**Gemini round 2** (`review/gemini2.py`, `gemini2_score.py`, `gemini2_results.md`).
- Input: target + 4 new same-track historicals (from 20261103), 96×96 at 4×, a 1-px circle
  at the label, the date drawn on each chip, lat/lon in the prompt.
- Five-way answer: fixed_structure / island / permanently_anchored_vessel / vessel /
  unsure. The first three mean delete.
- 231 labeled items. Each cell is predicted delete / keep / unsure:

| variant | other: delete labels (103) | other: vessel labels (27) |
|---|---|---|
| argon_base | 98 / 5 / 0 | 4 / 23 / 0 |
| argon_ext (extended prompt) | 98 / 5 / 0 | 2 / 25 / 0 |
| argon_fs6 / argon_fs10 (few-shot) | 95 / 5 / 3, 95 / 7 / 1 | 4 / 22 / 1, 3 / 24 / 0 |
| pro_base (gemini-3.1-pro-preview, high thinking) | 94 / 9 / 0 | 5 / 22 / 0 |
| pro_ext_fs6 / pro_ext_fs10 | 87 / 16 / 0, 89 / 14 / 0 | 2 / 25 / 0, 2 / 25 / 0 |

- The extended prompt describes islands, turbines, platforms and permanently anchored
  vessels: most stay in exactly the same spot, while anchored ones may move a little.
- Every variant deletes all 95 infra-group delete labels.
- Pro and few-shot don't help.
- The 2 vessels argon_ext deletes (other_0074, other_0111) are deleted by all 7 variants.
- Round 2 cost about $121. A launch mistake ran 6 variants twice; the duplicate runs agree
  on 223–231 of 231 items.

**Present-in-k rule** (`review/present_k.py`, `present_k.txt`). Delete if the object is
present (≥10 dB and NCC ≥0.7) in ≥k of the 4 new historicals.
- k≥3: 80/103 bad labels deleted, 2/27 vessels deleted. k≥4: 62/103 and 1/27.
- Combined with Gemini: argon_ext AND k≥2 gives 91/103 and 1/27.
- favyen chose argon_ext alone.

**Final pass.** argon_ext on all 2,257 other-group labels (`gemini2.py --all-other
argon_ext` → `review/gemini2_runs/all_other_argon_ext.jsonl`, about $24): fixed_structure
1,148, island 255, permanently_anchored 313, vessel 528, unsure 13.


5. dataset_v1/20261103
----------------------

Details are in `newhist/README.md`.

**Layout.**
- The windows, `metadata.json`, mask, target (`sentinel1`) and label layers are hardlinked
  from 20250521.
- `config.json` has a new `sentinel1_historical` layer (GRD VV_VH, time_offset -155d,
  duration 150d, period_duration 30d, max_matches 4, min_matches 1,
  per_period_mosaic_reverse_time_order true; no orbit_direction) and data sources in
  class_path form.
- `items.json` also carries `group_time_ranges`.

**Scene selection** (`newhist/select_hist.py`). This is our own prepare step, since rslearn
can't express "same track + random pick".
- Periods, most recent first: [t-35, t-5), [t-65, t-35), [t-95, t-65), [t-125, t-95),
  [t-155, t-125) days.
- A candidate pass is all GRD frames of one acquisition on the target's relative orbit
  (S1A `(abs-73)%175+1`, S1B `(abs-27)%175+1`). It qualifies if the union of its GRD
  footprints covers ≥98% of (target footprint ∩ window).
- One qualifying pass per period, random with an RNG seeded by md5(window id). Selection
  stops at 4 groups.
- Search: OEDatasets `sentinel-1-rtc` plus Copernicus OData for the GRD items, with OData as
  the fallback (21 windows).
- Checked per window: distinct scenes, same relative orbit as the target, each scene inside
  its period.

**Pixels.**
- AWS `sentinel-s1-l1c` windowed reads using rslearn's ingest warp, then rslearn
  materialize. 16 CPU Beaker jobs (IDs in `newhist/launch.log`).
- 14 scenes are missing on AWS. The 126 affected windows were re-selected without them.

**Result.**
- Historical groups per window: 4: 39,364 (98.58%), 3: 173, 2: 154, 1: 56, 0: 184 (kept,
  layer not completed).
- Group 0 is from 5–35 d for 98.7% of windows.

**Verification.**
- Where a group uses the same scene as 20250521: pixel corr median 0.9998, min 0.984 (near
  nodata edges).
- Label-patch NCC of target vs new hist1: median 0.511 vs 0.509 for the old hist1, so no
  alignment regression.

**Label edits** (`review/build_label_edits.py` → `review/label_edits.json`).
- Priority: favyen's label (infra / island / derelict → delete; vessel / moored → keep),
  then infra group → delete, then argon_ext (structure / island / anchored → delete;
  vessel / unsure / no answer → keep).
- Labels are matched to GeoJSON features by rounded pixel position.
- 2,590 features removed in 1,330 windows (875 infra group + 1,715 other); 543 kept. 225
  windows now have no labels.
- Each edited file is written and then `os.replace`d, so only the 20261103 link changes; the
  20250521 files were verified byte-identical.


6. Open items
-------------

- Train with `data/sentinel1_vessels/20261103_static_labels/config_{1,2}hist.yaml`.
  - Val numbers are not comparable to the June runs, since the val historicals changed.
    Evaluate the current production checkpoint on the 20261103 val set too.
  - The val set barely tests infrastructure false positives, so also re-run the 34 AOI
    scenes with same-track 5–35 d historicals.
- Prediction pipeline: restrict historical candidates to the target's relative orbit and
  search the same 30-day periods, before deploying a 20261103 model.
- Other findings from the investigation:
  - Skylight int appears to use a stale marine-infra map: Dogger detections 46–49 m from
    mapped turbines are not filtered.
  - The German Bight wind farm is not in the 2026-10-12 map.
  - The 0.7 threshold is the main driver of persistent false positives with the current
    model.
