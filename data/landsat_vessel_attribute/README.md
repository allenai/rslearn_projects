History:
- 20260330: For the initial Landsat vessel attribute prediction model, several model configs are
tested (see 20260330/). OlmoEarth-v1-Base performs the best, so it is copied to
`config.yaml` here for use in the Landsat vessel detection pipeline. The batch size is
reduced to 8 because 32 is too high for T4 GPU; for now it seems better to leave that
in the config rather than adding another environment override.
- 20260422: Hunter provided about 700K more examples. Total dataset size is about 1M now. It is a new
  group in the same rslearn dataset path.
- 2026_09_17: Compared L1 vs MSE loss for the regression heads (see 2026_09_17_l1_loss/).
  L1 performs better, so `config.yaml` now uses `loss_mode: l1` and points at the
  `2026_09_17_vessel_attribute_l1/landsat_l1` checkpoint.

The rslearn dataset is here:
- Dataset path: `/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_attribute/dataset_v1/20260330/`
