Training config for the 20261103 Sentinel-1 vessel model
=========================================================

`config_1hist_randhist.yaml` trains the deployed model (project `20261103_static_labels`,
run `data_20261103c3_model_20261107_satlaspretrain_unfreeze4_13_1hist_randhist`) on
`dataset_v1/20261103` with the labels from the 2026-11-07 cleanup. It is the same model as
`../config.yaml` (SatlasPretrain Swin-B, one historical image), except for one thing.
`image_1` lists all four same-track historicals (`sentinel1_historical`, `.1`, `.2`, `.3`),
so rslearn picks one at random per example, from 5 to 155 days before the target. Prediction
uses a single historical.

See `one_off_projects/2026_11_03_s1_static_labels/README.md` for how the dataset and labels
were made.


Comparison of training variants (2026-11-07)
--------------------------------------------

Before settling on this config we trained five variants on the second round of label
cleanup. That round predates the final Gemini check of every label, but it is the last
round where all five ran. Every model was evaluated with the most recent same-track
historical.

- 1hist: the previous production config on the new dataset, always using the most recent
  historical.
- 2hist: two historicals (the two most recent).
- ImageNet Swin: the 1hist model with an ImageNet-pretrained Swin-v2-B instead of
  SatlasPretrain.
- 1hist randhist / ImageNet Swin randhist: the 1hist and ImageNet Swin models trained with
  a random historical, as in `config_1hist_randhist.yaml`.

Validation: the score threshold is chosen per model to match the recall of the previous
production model at 0.7 / 0.8 / 0.9. The table shows threshold and precision.

| model | val mAP | @prod 0.7 (R .874) | @prod 0.8 (R .838) | @prod 0.9 (R .767) |
|---|---|---|---|---|
| previous production | 0.514 | 0.70, 0.775 | 0.80, 0.842 | 0.90, 0.912 |
| 1hist | 0.517 | 0.77, 0.824 | 0.87, 0.873 | 0.95, 0.923 |
| 2hist | 0.519 | 0.67, 0.827 | 0.78, 0.870 | 0.90, 0.922 |
| ImageNet Swin | 0.516 | 0.67, 0.813 | 0.77, 0.862 | 0.89, 0.912 |
| 1hist randhist | 0.517 | 0.67, 0.818 | 0.79, 0.868 | 0.91, 0.925 |
| ImageNet Swin randhist | 0.521 | 0.70, 0.827 | 0.81, 0.871 | 0.91, 0.917 |

34 scenes over offshore wind farms and platforms (Dogger Bank, German Bight, Stavanger).
Entries are detections remaining after the 200 m marine-infrastructure filter / persistent
detections (seen in at least 3 scenes within 100 m), at the matched thresholds:

| model | @prod 0.7 | @prod 0.8 | @prod 0.9 |
|---|---|---|---|
| previous production | 916 / 474 | 705 / 324 | 399 / 51 |
| 1hist | 476 / 39 | 390 / 18 | 281 / 0 |
| 2hist | 479 / 62 | 377 / 53 | 289 / 33 |
| ImageNet Swin | 385 / 17 | 321 / 6 | 237 / 0 |
| 1hist randhist | 409 / 6 | 321 / 0 | 240 / 0 |
| ImageNet Swin randhist | 367 / 4 | 284 / 4 | 227 / 1 |

Every model detected 24 or 25 of 25 AIS-correlated vessels (within 500 m) at the 0.7
and 0.8 equivalents.

Takeaways:

- Training with a random historical made the biggest difference: persistent detections at
  the 0.7 equivalent dropped from 39 to 6 (1hist) and from 17 to 4 (ImageNet Swin), with no
  loss of val precision or AIS recall.
- Two historicals did not help.
- ImageNet Swin randhist was slightly best. We kept 1hist randhist because it is nearly as
  good and keeps the SatlasPretrain backbone used in production.


Final model (2026-11-09)
------------------------

`config_1hist_randhist.yaml` was retrained on the final labels. The previous production
model was evaluated on the same validation set.

| model | val mAP | P / R at 0.7 | P / R at 0.8 | P / R at 0.9 |
|---|---|---|---|---|
| previous production | 0.527 | 0.740 / 0.919 | 0.812 / 0.891 | 0.892 / 0.826 |
| this model | 0.538 | 0.826 / 0.919 | 0.864 / 0.889 | 0.910 / 0.828 |

At the same threshold, the two models have the same recall, so no threshold remapping is
needed. Results on the 34 wind-farm scenes:

| model | @0.7 | @0.8 | @0.9 |
|---|---|---|---|
| previous production | 916 / 474 | 705 / 324 | 399 / 51 |
| this model | 299 / 3 | 265 / 0 | 220 / 0 |

- AIS-correlated vessels: this model detects 24/25 at 0.7 and 0.8 and 23/25 at 0.9. The
  previous model detects 25/24/24.
- The vessel missed at 0.7 and 0.8 is off SW Australia. This model scores it 0.62; the
  previous model scored it 0.76.
- The five false positives reported from Skylight (turbines and islands) all score below
  0.3.

The prediction pipeline's default threshold is 0.8. That keeps recall near the old model
at 0.7 (0.889 vs 0.919), gives zero persistent detections on the wind-farm scenes, and has
the same AIS recall as 0.7.
