Compare L1 vs MSE loss for the regression heads. Both configs are the same as the
production `../config.yaml` (SwinB, 128x128 input, trained on the 20260422 data), except
that the five regression heads (length, width, speed, heading_x, heading_y) set
`loss_mode` explicitly on `RegressionHead`. The ship type classification head is
unchanged.

- config_l1.yaml: `loss_mode: l1`, run `sentinel1_l1`.
- config_mse.yaml: `loss_mode: mse` (the default, so equivalent to the production
  config), run `sentinel1_mse`. This is the baseline re-trained in the same W&B project.

Both log to W&B project `2026_09_17_vessel_attribute_l1`, shared with the Sentinel-2 and
Landsat variants.

Result: L1 performs better, so `config_l1.yaml` (with the production dataset path) was
promoted to `../config.yaml` and the Sentinel-1 vessel detection pipeline now uses the
`2026_09_17_vessel_attribute_l1/sentinel1_l1` checkpoint.
