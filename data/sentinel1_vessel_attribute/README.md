History:
- 20260330: For the initial Sentinel-1 vessel attribute prediction model, several model configs are
tested (see 20260330/). The SwinB model with 128x128 input performs the best, so it is
copied to `config.yaml` here for use in the Sentinel-1 vessel detection pipeline.
- 20260422: Hunter provided about 700K more examples. Total dataset size is about 1M now. It is a new
  group in the same rslearn dataset path.

Note: from W&B it seems the last checkpoint provides a more balanced tradeoff across
all the attributes than the "best" checkpoint, so I copied `last.ckpt` to `best.ckpt`.

The rslearn dataset is here:
- Dataset path: `/weka/dfive-default/rslearn-eai/datasets/sentinel1_vessel_attribute/dataset_v1/20260330/`
