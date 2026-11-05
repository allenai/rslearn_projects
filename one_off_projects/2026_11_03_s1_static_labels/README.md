Sentinel-1 vessel dataset 20261103: static-label cleanup + same-track historicals
=================================================================================

November 2026. Skylight users reported that the Sentinel-1 vessel detection model
(`data/sentinel1_vessels/config.yaml`, the 1-historical model) flags fixed
infrastructure as vessels. We found the cause and built a new dataset version,
`dataset_v1/20261103`, to retrain on. The full write-up, with numbers, scripts and
paths, is in `AGENT_DETAILED_README.md`.


What we found
-------------

- The reported detections reproduce exactly. The historical image is recent and
  aligned (0–2 px), and the objects are visible in it.
- The model uses the historical image only weakly. An object that is also in the
  historical still scores ≥0.7 about half the time, so the lowered 0.7 threshold lets
  persistent objects through.
- The training labels are the main cause. In `dataset_v1/20250521`, 3,133 training labels
  (5.7%) sit on an object that looks the same in the historical image:
  - 876 are within 100 m of a mapped platform or wind turbine;
  - 2,257 others are islands, unmapped structures and permanently anchored ships.
  All are labeled "vessel", so the model learned that "present in the historical" does
  not mean "not a vessel".
- The old historicals were chosen by orbit direction only, so 15% (hist1) and 23% (hist2)
  of these labels had a historical from a different track (different viewing geometry).
- A 2-historical model gives no gain at matched recall.


What changed in dataset_v1/20261103
-----------------------------------

Path: `/weka/dfive-default/rslearn-eai/datasets/sentinel1_vessels/dataset_v1/20261103`.
It has the same 39,931 windows, target images and masks as 20250521 (hardlinked; 20250521
is unchanged).

1. Historicals: one `sentinel1_historical` layer with up to 4 item groups, most recent
   first. Each group is a pass on the target's relative orbit (same track) that covers the
   window. The window is split into five 30-day periods before the target
   (5–35, 35–65, 65–95, 95–125 and 125–155 days), with one random qualifying pass per
   period and empty periods skipped. In rslearn terms: time_offset -155d, duration 150d,
   period_duration 30d, max_matches 4, min_matches 1.
   - Windows with 4 groups: 98.58%; 3: 0.43%; 2: 0.39%; 1: 0.14%; 0: 0.46%. Windows with 0
     are kept, but the layer is not completed, so training skips them.
   - Group 0 (`sentinel1_historical`, what the 1-hist model reads) comes from 5–35 days
     before the target for 98.7% of windows (it was 30–90 days in 20250521).
2. Labels: 2,590 of the 3,133 static train labels and 293 of the 348 static val labels are
   deleted. The windows stay in the dataset as hard negatives. 61,655 labels → 58,772.
   - The 876 labels near mapped infrastructure are deleted (875; favyen marked one as a
     moored vessel).
   - For the other 2,257, favyen's manual labels (142 of them reviewed) take priority. Otherwise
     Gemini (`gemini-4-argon`, five images: target + 4 historicals) decides: fixed structure,
     island and permanently anchored vessel are deleted; vessel and unsure are kept. On
     favyen's labels it deleted 98/103 bad labels and 2/27 correct vessel labels.
   - Val gets the same treatment, minus the manual review: 109 labels near mapped
     infrastructure deleted, plus 184 of the other 239 that Gemini flags. Val had the same
     static-label rate as train (5.7%), so the old val mAP partly rewarded detecting
     infrastructure.


Training
--------

Configs: `data/sentinel1_vessels/20261103_static_labels/config_1hist.yaml` and
`config_2hist.yaml`. They are the 20260626 configs pointed at 20261103. The layer names are
unchanged: `sentinel1_historical` is now group 0 (most recent same-track pass) and
`sentinel1_historical.1` is group 1.

Note for deployment: the prediction pipeline (`rslp/sentinel1_vessels/predict_pipeline.py`)
still picks the historical from [t-60d, t-30d] by orbit direction only. It must be updated
(same relative orbit, the same 30-day periods) before a model trained on 20261103 is
deployed.


Cleanup round 2 and retrain (2026-11-05)
----------------------------------------

The 20261103 labels were cleaned a second time, in place:
- 2,248 labels that look static against the most recent same-track historical (VH or VV
  peak/NCC test) but were not candidates on 11-03 were checked with Gemini (merged prompt:
  infrastructure covers turbines, platforms and anchored structures; no separate anchored
  class). 1,322 were deleted.
- The 336 labels deleted on 11-03 as "permanently anchored vessel" were re-checked the same
  way; 161 were restored from 20250521.
- Labels: 61,655 in 20250521, 57,611 in 20261103 now. On a random 100 of the Gemini deletions,
  65 of 66 that favyen labeled were infrastructure.

The earlier 1hist/2hist runs on 20261103 were stopped. Five new runs (run names
`data_20261103c2_model_20261105_*`, so they do not resume the old checkpoints):
- `config_1hist.yaml`, `config_2hist.yaml`: as before.
- `config_cmp_swin_imagenet.yaml`: 1hist, but a torchvision ImageNet Swin-v2-B encoder
  (4-channel input, randomly initialized patch embedding, no freezing) with the same SAR
  normalization.
- `config_1hist_randhist.yaml`, `config_cmp_swin_imagenet_randhist.yaml`: the historical is
  picked at random from the up-to-4 same-track historicals (rslearn picks one completed entry
  of `layers` per example; train re-picks each time, val/test fixed per window), instead of
  always the most recent one.
