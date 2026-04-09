For the initial Landsat vessel attribute prediction model, several model configs are
tested (see 20260330/). OlmoEarth-v1-Base performs the best, so it is copied to
`config.yaml` here for use in the Landsat vessel detection pipeline. The batch size is
reduced to 8 because 32 is too high for T4 GPU; for now it seems better to leave that
in the config rather than adding another environment override.

The rslearn dataset is here:
- Dataset path: `/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_attribute/dataset_v1/20260330/`
