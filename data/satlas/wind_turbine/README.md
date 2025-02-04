`config.json` and `config.yaml` contain the active dataset and model configurations for
the wind turbine point detection model.

It uses six monthly Sentinel-2 L1C mosaics, which can be spread over up to nine months
(in case some months don't have enough cloud-free images to form a mosaic).

`config_azure.json` is for testing a model that inputs Sentinel-2 L2A + Sentinel-1
vv+vh sourced from Microsoft Azure.
