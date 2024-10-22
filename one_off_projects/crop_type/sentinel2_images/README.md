This project downloads Sentinel-2 images for our crop type mapping
collaboration with Yi-Chia Chang et al at UIUC.

First, use torchgeo to download the crop type labels:

    from torchgeo.datasets import CDL, EuroCrops, NCCM, SouthAmericaSoybean
    CDL(paths='./data/cdl/', download=True, checksum=True, years=[2023, 2022, 2021, 2020, 2019, 2018, 2017])
    EuroCrops(paths='./data/eurocrops/', download=True, checksum=True)
    NCCM(paths="./data/nccm/", download=True, checksum=True)
    SouthAmericaSoybean(paths="./data/sas/", download=True, checksum=True)

We use those labels to identify which 256x256 10 m/pixel tiles in WebMercator
projection we want to download. Note that this resolution does not conform to
any standard WebMercator zoom level.

    python 1_identify_tiles.py

Then initialize an rslearn dataset using the computed tiles. The dataset path
is hardcoded.

    python 2_populate_windows.py /path/to/crop_type_tiles_cdl.json cdl

`config.json` is the dataset configuration that should be placed in the dataset
root directory.

Then use rslearn.main to prepare, ingest, and materialize the images.
After materialization, upload it to R2. The group to upload is hardcoded.

    python 4_upload_to_r2_v2.py

This script creates an index file that should also be manually uploaded to R2.
The `download_script.py` is an example for downloading the files from R2 in
parallel.


AgriFieldNet and SACT
---------------------

These have small 256x256 labels instead of much larger label images. Because
they do not align with the 256x256 10 m/pixel WebMercator images we are
downloading, they need separate procedure to initialize the windows in the
rslearn dataset.

    python 2_other_windows.py

This will create one window for each of three different calendar months.
Although the rslearn dataset configuration says to get the least cloudy
Sentinel-2 scenes for each month, that scene may still be cloudy in any one
particular tile.
So we have another script to pick the least cloudy images:

    python pick_least_cloud.py

Both the scripts here have lots of hardcoded paths, and we did not upload to R2
and instead the outputs from the latter script were used directly.
