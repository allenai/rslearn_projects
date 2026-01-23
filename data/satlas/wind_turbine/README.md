`config.json` and `config.yaml` contain the active dataset and model configurations for
the wind turbine point detection model.

It uses six monthly Sentinel-2 L1C mosaics, which can be spread over up to nine months
(in case some months don't have enough cloud-free images to form a mosaic).

`config_azure.json` is for testing a model that inputs Sentinel-2 L2A + Sentinel-1
vv+vh sourced from Microsoft Azure, along with Landsat from AWS bucket.


Dataset Versions
----------------

- Sentinel-2 L1C (config.json): `rslearn-eai/datasets/wind_turbine/dataset_v1/20241002/`
- Old Azure with Sentinel-2 L2A and Sentinel-1: `rslearn-eai/datasets/wind_turbine/dataset_v1/20241212/`
- New Azure with Sentinel-2 L2A, Sentinel-1, and Landsat: `rslearn-eai/datasets/wind_turbine/dataset_v1/20250605/`
- Fixed naip group: `rslearn-eai/datasets/wind_turbine/dataset_v1/20260122/`
    - It looks like the coordinates of the labels in the naip group were not
      re-projected properly when this dataset was first converted from the multisat
      version -- they were still in WebMercator coordinates and so the model was
      trained not to detect anything in all of the naip windows (which were originally
      created by training and applying an object detector with high-res NAIP images,
      and using those outputs to supervise Sentinel-2 training). So this new version is
      the same as 20250605 but with the labels fixed. The script to fix the labels is
      `one_off_projects/2026_01_22_fix_wind_turbine_dataset/run.py`.
