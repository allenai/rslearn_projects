# Live Fuel Moisture Content (LFMC) Estimation

The original LFMC work is based on the Galileo model (more details can be found in Patrick J's [paper](https://arxiv.org/abs/2506.20132)), with data preparation relying on Google Earth Engine. Here, we switch to rslearn for data preparation, with two options for the model, SatlasPretrain and OlmoEarth (the latter is not open source yet).

Right now, only the following modalities are supported: Sentinel2, Sentinel1, SRTM, and location. We will need to add ERA5 and maybe TerraClimate in the future -- the ablation study of the paper (Table 9) indicates that adding ERA5 and TerraClimate helps reduce the error in LFMC estimation.

### 1. Prepare Labels

Steps 1-3 cover creating an rslearn dataset with the LFMC data. Alternatively you can
download and extract the dataset from the tar URL below and skip to step 4. This
dataset includes landcover and elevation tags on the windows that were obtained
separately, to support breakdown of results by those categories.

- https://storage.googleapis.com/ai2-rslearn-projects-data/datasets/lfmc/20250626.tar

Run the following command to prepare labels (note that this script requires pandas to
be additionally installed):
```
python -m rslp.lfmc.prepare_labels --csv_path /tmp/lfmc-labels.csv
```

This file is the processed LFMC ground-truth data, which includes metadata like `latitude`, `longitude`, `sampling_date`, `site_name`, `lfmc_value`, `state_region`, and `country`. We further cut off the LFMC value by 302 which is the 99.9% value.

### 2. Create Windows

Run the following commands to create windows (note that this script requires pandas to
be additionally installed):
```
export DATASET_PATH=/path/to/dataset
mkdir -p $DATASET_PATH
cp data/lfmc/config.json $DATASET_PATH/config.json
python -m rslp.lfmc.create_windows --csv_path /tmp/lfmc-labels.csv --ds_path $DATASET_PATH --window_size 32
```

### 3. Prepare/Ingest/Materialize Windows

Run the command to prepare/ingest/materialize groundtruth windows (the ingestion is mainly for SRTM):
```
rslearn dataset prepare --root $DATASET_PATH --workers 64 --retry-max-attempts 8 --retry-backoff-seconds 60
rslearn dataset ingest --root $DATASET_PATH --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
rslearn dataset materialize --root $DATASET_PATH --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
```

Note that to run these commands, we will need to set the environmental variables of `NASA_EARTHDATA_USERNAME` and `NASA_EARTHDATA_PASSWORD` ([Link](https://urs.earthdata.nasa.gov/)) for accessing SRTM data.

### 4. Finetune SatlasPretrain

Now train the SatlasPretrain model. The model configuration here only uses the
Sentinel-2 images. Note that you will need to setup the RSLP_PREFIX. This model
achieves 12.6 L1 error, and uses only Sentinel-2 images.

```
python -m rslp.rslearn_main model fit --config data/lfmc/config_satlaspretrain.yaml --data.init_args.path=$DATASET_PATH
```

If you don't want to train the model, you can get the checkpoint:

```
mkdir -p project_data/projects/lfmc/lfmc_satlaspretrain_s2_ts_ws32_03/checkpoints
wget https://storage.googleapis.com/ai2-rslearn-projects-data/projects/lfmc/lfmc_satlaspretrain_s2_ts_ws32_03/checkpoints/best.ckpt -O project_data/projects/lfmc/lfmc_satlaspretrain_s2_ts_ws32_03/checkpoints/best.ckpt
```

### 5. Make Prediction

Create a fresh dataset and a window in it for prediction. Like with training, the
windows should be 32x32 pixels (in UTM, 10 m/pixel) centered at the point that you want
a prediction for. In this example, we instead create a grid of 32x32 windows covering a
target region at two timestamps, and predict LFMC value for each window. The window's
time range is a point in time and the prediction will be for the 30-day period ending
at that time.

Note that here we only get Sentinel-2 images since that is all the SatlasPretrain model
above uses. The ingest step should show zero ingest jobs since we directly materialize
from Sentinel-2 COGs on Microsoft Planetary Computer.

```
export INFERENCE_PATH=/path/to/inference_dataset
mkdir -p $INFERENCE_PATH
cp data/lfmc/config.json $INFERENCE_PATH/config.json
rslearn dataset add_windows --root $INFERENCE_PATH --group predict --box=-121.55,37.18,-121.52,37.21 --src_crs EPSG:4326 --utm --resolution 10 --start "2025-03-01T00:00:00Z" --end "2025-03-01T00:00:00Z" --grid_size 32
rslearn dataset add_windows --root $INFERENCE_PATH --group predict --box=-121.55,37.18,-121.52,37.21 --src_crs EPSG:4326 --utm --resolution 10 --start "2025-08-01T00:00:00Z" --end "2025-08-01T00:00:00Z" --grid_size 32
rslearn dataset prepare --root $INFERENCE_PATH --group predict --workers 64 --retry-max-attempts 8 --retry-backoff-seconds 60 --disabled-layers sentinel1,srtm
rslearn dataset ingest --root $INFERENCE_PATH --group predict --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60 --disabled-layers sentinel1,srtm
rslearn dataset materialize --root $INFERENCE_PATH --group predict --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60 --disabled-layers sentinel1,srtm
python -m rslp.rslearn_main model predict --config data/lfmc/config_satlaspretrain.yaml --data.init_args.path=$INFERENCE_PATH --load_best=true
```

### 6. Finetune OlmoEarth

In `data/helios/v2_lfmc/README.md` there are Ai2-internal instructions for fine-tuning
the OlmoEarth model on Helios. These commands launch Beaker jobs and use the model that
is not open source yet, so it is Ai2-specific. We plan to release the model in November
2025, after which these instructions should be updated to not involve Beaker. Also the
`data/helios/` should be for Helios comparison experiments, while the best model
configuration for a specific project like this should live in `data/lfmc/config.yaml`.
