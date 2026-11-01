Weekly marine infrastructure models with up to 12 training images and a temporal crop (2026-10-31).

- Dataset: `/weka/dfive-default/rslearn-eai/datasets/marine_infra/dataset_v1/20261031_weekly12_256/`.
  Train tiles are the train tiles of `../weekly_20261030/` with up to 12 weekly Sentinel-2 L2A images over
  the same 126-day window (`dataset_config.json`: `max_matches: 12`, chronological order, so the newest
  images are the same scenes as in the 6-image dataset). Val tiles are copied unchanged from
  `20261030_weekly_256` (6 images), so val metrics compare directly with `../weekly_20261030/`.
- Training: `rslp.satlas.transforms.RandomConsecutiveTimesteps` keeps a random run of 6 consecutive
  images per training sample (tiles with 6 or fewer keep all). Satlas loads all item groups and trains
  only on tiles with option `num_images_ge6` (SimpleTimeSeries needs exactly 6 images).
- Models: `satlas.yaml`, `olmoearth_v1_2_{small,base}.yaml` and `olmoearth_v1_2_{small,base}_llrd.yaml`,
  otherwise identical to the same-named configs in `../weekly_20261030/`.
