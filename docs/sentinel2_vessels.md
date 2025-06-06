Sentinel-2 Vessel Detection
---------------------------

The Sentinel-2 vessel detection model detects ships in Sentinel-2 L1C scenes. We use
L1C instead of L2A since L1C scenes are released with a lower latency, and latency is
important for [Skylight](https://www.skylight.global/) (which is the primary use of
this model within Ai2).

It is trained on a dataset consisting of 43,443 image patches (ranging from 300x300 to
1000x1000) with 37,145 ship labels. See [our paper](https://arxiv.org/pdf/2312.03207)
for more details about the model and dataset.

![Image showing a Sentinel-2 image with predicted positions of ships from the model overlayed.](./images/sentinel2_vessels/prediction.png)


Inference
---------

First, download the model checkpoint to the `RSLP_PREFIX` directory.

    cd rslearn_projects
    mkdir -p project_data/projects/sentinel2_vessels/data_20240927_satlaspretrain_patch512_00/checkpoints/
    mkdir -p project_data/projects/sentinel2_vessel_attribute/data_20250205_regress_00/checkpoints/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel2_vessels/data_20250213_02_all_bands/checkpoints/best.ckpt -O project_data/projects/sentinel2_vessels/data_20250213_02_all_bands/checkpoints/best.ckpt
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel2_vessel_attribute/data_20250205_regress_00/checkpoints/best.ckpt -O project_data/projects/sentinel2_vessel_attribute/data_20250205_regress_00/checkpoints/best.ckpt

The easiest way to apply the model is using the prediction pipeline in
`rslp/sentinel2_vessels/predict_pipeline.py`. It accepts a Sentinel-2 scene ID and
automatically downloads the scene images from a
[public Google Cloud Storage bucket](https://cloud.google.com/storage/docs/public-datasets/sentinel-2).

    mkdir output_crops
    mkdir scratch_dir
    python -m rslp.main sentinel2_vessels predict --tasks '[{"scene_id": "S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425", "geojson_path": "out.geojson", "crop_path": "output_crops/"}]' --scratch_path scratch_dir/
    qgis out.geojson scratch_dir/windows/default/default/layers/sentinel2/R_G_B/geotiff.tif

Then, `out.geojson` will contain a GeoJSON of detected ships while `output_crops` will
contain corresponding crops centered around those ships (showing the RGB B4/B3/B2
bands).


Training
--------

First, download the training dataset:

    cd rslearn_projects
    mkdir -p project_data/datasets/sentinel2_vessels/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/sentinel2_vessels/sentinel2_vessels.tar -O project_data/datasets/sentinel2_vessels.tar
    tar xvf project_data/datasets/sentinel2_vessels.tar --directory project_data/datasets/sentinel2_vessels/

It is an rslearn dataset consisting of window folders like
`windows/sargassum_train/1186117_1897173_158907/`. Inside each window folder:

- `layers/sentinel2/` contains different Sentinel-2 bands used by the model, such as
  `layers/sentinel2/R_G_B/image.png`.
- `layers/label/data.geojson` contains the positions of ships. These are offset from
  the bounds of the window which are in `metadata.json`, so subtract the window's
  bounds to get pixel coordinates relative to the image.

Use the command below to train the model. Note that Weights & Biases is needed. You can
disable W&B with `--no_log true` but then it may be difficult to track the metrics.

    python -m rslp.rslearn_main model fit --config data/sentinel2_vessels/config.yaml --data.init_args.path project_data/datasets/sentinel2_vessels/

To visualize outputs on the validation set:

    mkdir vis
    python -m rslp.rslearn_main model test --config data/sentinel2_vessels/config.yaml --data.init_args.path project_data/datasets/sentinel2_vessels/ --model.init_args.visualize_dir vis/ --load_best true


Model Version History
---------------------

The version names correspond to the `rslp_experiment` field in the model configuration
file (`data/sentinel2_vessels/config.yaml`).

- `data_20250213_02_all_bands`: Train on all bands instead of just RGB. Note that it
  uses B01-B12 instead of TCI so it needs "harmonization" (subtracting 1000 from new
  Sentinel-2 products).
- `data_20240213_01_add_freezing_and_fix_fpn_restore`: Freeze the pre-trained model for
  the first few epochs before unfreezing.
- `data_20240213_00`: Some of the windows contained blank images. I re-ingested the
  dataset and the issue seems to be fixed. The model is re-trained.
- `data_20240927_satlaspretrain_patch512_00`: initial model.


Model Performance
-----------------

### data_20250213_02_all_bands

- Selected threshold: 0.8
- Results on validation set (split1, split7, sargassum_val)
  - Precision: 77.2%
  - Recall: 78.6%

### data_20240213_01_add_freezing_and_fix_fpn_restore

- Selected threshold: 0.8
- Results on validation set (split1, split7, sargassum_val)
  - Precision: 78.0%
  - Recall: 77.6%
- Note it should be 20250213 but there is typo.

Docker Container with FastAPI
-----------------------------

We also have a Docker container that exposes a FastAPI interface to apply vessel
detection on Sentinel-2 scenes. This section explains how to setup the API.

### Run the Docker container

The Docker container does not contain the model weights. Instead, it expects the model
weights to be present in a directory based on the `RSLP_PREFIX` environment variable.
So download the model checkpoint:

    mkdir -p project_data/projects/sentinel2_vessels/data_20250213_02_all_bands/checkpoints/
    mkdir -p project_data/projects/sentinel2_vessel_attribute/data_20250205_regress_00/checkpoints/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel2_vessels/data_20250213_02_all_bands/checkpoints/best.ckpt -O project_data/projects/sentinel2_vessels/data_20250213_02_all_bands/checkpoints/best.ckpt
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel2_vessel_attribute/data_20250205_regress_00/checkpoints/best.ckpt -O project_data/projects/sentinel2_vessel_attribute/data_20250205_regress_00/checkpoints/best.ckpt

Run the container:

```bash
export SENTINEL2_PORT=5555
docker run \
    --rm -p $SENTINEL2_PORT:$SENTINEL2_PORT \
    -e RSLP_PREFIX=/project_data \
    -e SENTINEL2_PORT=$SENTINEL2_PORT \
    -v $PWD/project_data/:/project_data/ \
    --shm-size=15g \
    --gpus all \
    ghcr.io/allenai/sentinel2-vessel-detection:sentinel2_vessels_v0.0.6
```

### Auto Documentation

This API has enabled Swagger UI (`http://<your_address>:<port_number>/docs`) and ReDoc (`http://<your_address>:<port_number>/redoc`).

### Making Requests

Process a scene by its Sentinel-2 scene ID. Note that the crop path is optional.

```bash
curl -X POST http://localhost:${SENTINEL2_PORT}/detections -H "Content-Type: application/json" -d '{"scene_id": "S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425", "crop_path": "crops/"}'
```

The API will respond with the vessel detection results in JSON format.

Alternatively, process the scene by providing the paths to the image assets. The paths
can be URIs but must be accessible from the Docker container.

```bash
curl -X POST http://localhost:${SENTINEL2_PORT}/detections -H "Content-Type: application/json" -d '{"image_files": [{"bands": ["B08"], "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B08.jp2"}, ...]}'
```

These bands must be provided. In this case the scene must be processed with processing baseline 04.00 or later (i.e. has N0400 or higher in the scene ID) since it is assumed to be the newer type where the same intensity has 1000 higher pixel value (we subtract 1000 in `data/sentinel2_vessels/config_predict_local_files.json`, see [GEE Harmonized Sentinel-2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) for details).

- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B10
- B11
- B12
- B8A

### Docker Container Version History

- v0.0.6: enable Pytorch Lightning environment variable parsing to allow disabling progress bar via environment variable.
- v0.0.5: add Prometheus metrics.
- v0.0.4: fix bug predicting attributes in scenes with zero vessel detections.
- v0.0.3: fix bug with image_files execution.
- v0.0.2: add attribute prediction (`data_20250205_regress_00`) and use model `data_20250213_02_all_bands`.
- v0.0.1: initial version. It uses model `data_20240213_01_add_freezing_and_fix_fpn_restore`.


Vessel Attribute Prediction
---------------------------

The vessel attribute prediction model predicts the vessel type, length, width, speed,
and heading of each detected vessel. The predicted values are available under the
"attributes" key of the JSON or GeoJSON vessel object.
