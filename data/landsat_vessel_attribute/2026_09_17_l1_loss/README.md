Compare L1 vs MSE loss for the regression heads. Both configs are the same as the
production `../config.yaml` (OlmoEarth-v1-Base with freeze/unfreeze, trained on the
20260422 data), except that the five regression heads (length, width, speed, heading_x,
heading_y) set `loss_mode` explicitly on `RegressionHead`. The ship type classification
head is unchanged.

- config_l1.yaml: `loss_mode: l1`, run `landsat_l1`.
- config_mse.yaml: `loss_mode: mse` (the default, so equivalent to the production
  config), run `landsat_mse`. This is the baseline re-trained in the same W&B project.

Both log to W&B project `2026_09_17_vessel_attribute_l1`, shared with the Sentinel-1 and
Sentinel-2 variants.

Result: L1 performs better, so `config_l1.yaml` was promoted to `../config.yaml` and the
Landsat vessel detection pipeline now uses the
`2026_09_17_vessel_attribute_l1/landsat_l1` checkpoint.
