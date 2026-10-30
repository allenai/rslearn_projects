Weekly 6-image marine infrastructure dataset and models (2026-10-30).

- Dataset: `/weka/dfive-default/rslearn-eai/datasets/marine_infra/dataset_v1/20261030_weekly_256/`,
  built from the 20241210 windows (UTM, 10 m/pixel). Each window keeps its original end time with
  a 126-day time range, and is split into 256x256 tiles. Sentinel-2 L2A comes from OlmoEarth Datasets
  (`dataset_config.json`): MOSAIC with `period_duration: 7d`, `max_matches: 6` and scene cloud cover
  < 50%, so it takes the six most recent weeks that have imagery. Val tiles with fewer than six images
  are dropped.
- Models (crop 128, batch 32, 32768 samples/epoch, lr 1e-4, encoder frozen for 5 epochs then
  unfrozen at lr/10, best checkpoint by val mAP):
  - `satlas.yaml`: SatlasPretrain Swin-B multi-image init (as in production), all six images required.
  - `olmoearth_v1_2_small.yaml`, `olmoearth_v1_2_base.yaml`, `olmoearth_v1_3_base.yaml`: OlmoEarth
    patch size 8 with real timestamps, Faster R-CNN head, all available weekly images.
