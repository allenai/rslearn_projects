For the initial Sentinel-1 vessel attribute prediction model, several model configs are
tested (see 20260330/). The SwinB model with 128x128 input performs the best, so it is
copied to `config.yaml` here for use in the Sentinel-1 vessel detection pipeline.

The rslearn dataset is here:
- Dataset path: `/weka/dfive-default/rslearn-eai/datasets/sentinel1_vessel_attribute/dataset_v1/20260330/`
