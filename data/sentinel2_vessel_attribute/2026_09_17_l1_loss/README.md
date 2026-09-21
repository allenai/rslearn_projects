Compare L1 vs MSE loss for the regression heads. Both configs are the same as the
production `../config.yaml` (SatlasPretrain SwinB), except that the five regression
heads (length, width, speed, heading_x, heading_y) set `loss_mode` explicitly on
`RegressionHead`. The ship type classification head is unchanged.

- config_l1.yaml: `loss_mode: l1`, run `sentinel2_l1`.
- config_mse.yaml: `loss_mode: mse` (the default, so equivalent to the production
  config), run `sentinel2_mse`. This is the baseline re-trained in the same W&B project.

The dataset path is also switched from the `gs://` URL to the Weka mirror so it trains
from Weka like the Sentinel-1 and Landsat variants.

Both log to W&B project `2026_09_17_vessel_attribute_l1`, shared with the Sentinel-1 and
Landsat variants. Note that the Sentinel-2 configs use project name
`2026_09_17_vessel_attribute_l1_01`, so the checkpoints live there rather than under
`2026_09_17_vessel_attribute_l1`.

Result: L1 performs better, so the `loss_mode: l1` heads were promoted to
`../config.yaml` (keeping the `gs://` dataset path and `num_workers: 64` from the
production config) and the Sentinel-2 vessel detection pipeline now uses the
`2026_09_17_vessel_attribute_l1_01/sentinel2_l1` checkpoint. This replaces the
`sentinel2_vessel_attribute/data_20250205_regress_00` model.
