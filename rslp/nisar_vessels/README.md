# NISAR Vessel Detection

Vessel detection models trained on NISAR L2 GCOV imagery, with labels from an
OlmoEarth Studio annotation project.

The dataset uses the `NisarL2Gcov` data source from olmoearth_run to materialize the
HHHH and HVHV backscatter bands (nearly all NISAR L-band science acquisitions are
dual-pol H-transmit, so these bands are available in almost all granules; granules
lacking them are skipped automatically). Vessel labels are point features in the
`label` vector layer with `category=vessel`, same as the sentinel1_vessels and
sentinel2_vessels projects.

## 1. Create the dataset

Tasks in the Studio project with status `reviewed` or `to_be_reviewed` each become one
rslearn window (bounds from the task geometry in UTM at 10 m/pixel, time range from
the task start/end times plus a one-minute buffer on each side, since the task
timestamps equal the granule acquisition start time and the window would otherwise
have zero duration and never match). Non-rejected annotations become vessel points in the
`label` vector layer. Windows are assigned to a `train` or `val` group (~90/10) by
hashing the window name. The dataset config (`data/nisar_vessels/config.json`) is
copied into the dataset root automatically.

Some crops were labeled by several annotators, each in their own task (the tasks
share the `crop_uid` attribute). If all of them labeled the same number of vessels,
only the first task becomes a window; if the counts differ, the crop is skipped.

Run from the rslearn_projects root:

    export STUDIO_API_KEY=...
    python -m rslp.nisar_vessels.scripts.create_dataset \
        --project_id c927e5cb-734b-4323-8cad-7f224b3e850d \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/dataset_20261013/

The current configs train on `dataset_20261014`, built from `dataset_20261013` with the
same windows, labels and splits but with each window's imagery taken from exactly the
granule the annotator labeled (the task's `frame_id` attribute) instead of whichever
granule `prepare` matched first; that differed for 88 of 768 windows. Train windows
labeled on a 40 MHz granule (MODE 4005) also got a copy, named `<window>_20mhz`, on a
20 MHz (MODE 2005) granule from the same pass when one covers the window and all its
labels (49 windows). Windows with no imagery or all-nodata imagery were dropped. The
scripts are in `/weka/dfive-default/agent/favyen/2026-10-14-favyen-nisar-frame-granule/`
(`find_granules.py`, `build_dataset_20261014.py`); `create_dataset` does not do this yet.

## 2. Prepare and materialize

This looks up matching NISAR L2 GCOV granules and writes the HHHH/HVHV rasters for
each window. It requires:

- olmoearth_run (main) installed with its `runner` extra, since the dataset config
  references `olmoearth_run...nisar.l2_gcov.NisarL2Gcov`.
- NASA Earthdata credentials in `EARTHDATA_CREDENTIALS` (JSON with `username` and
  `password`) to download the granules from ASF.

The NISAR layer is configured with `ingest: false`, so there is no separate
`rslearn dataset ingest` step; materialize reads directly from the granules.

    export EARTHDATA_CREDENTIALS='{"username": "...", "password": "..."}'
    rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/dataset_20261013/ --workers 32
    rslearn dataset materialize --root /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/dataset_20261013/ --workers 32

Note that granules that don't have the HHHH/HVHV bands (i.e., acquisitions that are
not dual-pol H-transmit) are skipped during prepare, so afterwards it is worth
checking for windows that failed to match any granule.

## 3. Train models

There are four model configs, all using a Faster R-CNN detection head on the two
NISAR bands (converted to decibels):

- `data/nisar_vessels/config_imagenet.yaml`: ImageNet-pretrained SwinB + FPN.
- `data/nisar_vessels/config_satlas.yaml`: SatlasPretrain Sentinel-1 SwinB + FPN
  (NISAR HH/HV stand in for Sentinel-1 vv/vh).
- `data/nisar_vessels/config_olmoearth_tiny.yaml`: OlmoEarth v1.2-Tiny, passing the
  NISAR bands as the `sentinel1` modality.
- `data/nisar_vessels/config_olmoearth_base.yaml`: same but OlmoEarth v1.2-Base.

Launch on Beaker via the common launcher (the W&B project/run names come from
`project_name`/`run_name` in each config):

    python -m rslp.main common beaker_train \
        --config_path data/nisar_vessels/config_olmoearth_tiny.yaml \
        --image_name YOUR_BEAKER_IMAGE \
        --cluster+=ai2/jupiter \
        --cluster+=ai2/ceres \
        --priority urgent \
        --gpus 1 \
        --shared_memory 256GiB \
        --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}'

Repeat with the other config paths to train the other models.

To train locally instead:

    rslearn model fit \
        --config data/nisar_vessels/config_olmoearth_tiny.yaml \
        --data.init_args.path /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/dataset_20261014/

## 4. Prediction windows (e.g. for mining hard negatives)

`create_predict_windows` searches the olmoearth_datasets API (requires the
`OEDATASETS_API_URL` and `DATASETS_API_TOKEN` environment variables) for dual-pol
H-transmit NISAR L2 GCOV scenes acquired in a time range (the only mode with the
HHHH/HVHV bands), and creates one unlabeled window per scene
at a random location within the scene footprint (deterministic per scene, so
re-running does not move existing windows). The windows go into the `predict` group
by default (matching `predict_config` in the model configs); use `--group` to
override.

Note the scene volume: globally there are roughly 1500 dual-pol scenes per day, so
use `--max_scenes` to randomly subsample. For example, 1000 windows of 2048x2048
from 2026-01-01 through 2026-10-12, in a separate dataset (its config.json must exist
first):

    mkdir -p /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/predict_20261013/
    cp data/nisar_vessels/config.json /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/predict_20261013/
    python -m rslp.nisar_vessels.scripts.create_predict_windows \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/predict_20261013/ \
        --start_time 2026-01-01T00:00:00Z \
        --end_time 2026-10-13T00:00:00Z \
        --window_size 2048 \
        --max_scenes 1000

Then prepare/materialize just that group, and run prediction (optionally lowering
the confidence threshold to favor recall, so false positives can be reviewed and
added back to the dataset as negative examples):

    rslearn dataset prepare --root <ds_path> --group predict --workers 32
    rslearn dataset materialize --root <ds_path> --group predict --workers 32
    rslearn model predict \
        --config data/nisar_vessels/config_satlas.yaml \
        --data.init_args.path <ds_path> \
        --data.init_args.task.init_args.tasks.detect.init_args.score_threshold=0.05

This uses the `data_20261014_satlas_00` checkpoint named in the config. At
0.05 the output keeps nearly everything the detector emits (Faster R-CNN drops boxes
below 0.05 itself), so a stricter threshold can be applied afterwards; on the
dataset_20261014 val split, 0.21 gave 95% recall at 0.66 precision. Predictions are
written to the `output` layer of each window.

## 5. Prediction service

`rslp/nisar_vessels/api_main.py` is a FastAPI server that runs the detector over one
granule per request. It is deployed as a sidecar next to the Skylight sat service, which
downloads the granule to a shared volume and posts its path:

    curl -X POST localhost:5555/detections -H 'Content-Type: application/json' -d '{
        "h5_path": "/shared/NISAR_L2_GCOV_..._20260101T000312_20260101T000347.h5",
        "crop_path": "/shared/crops",
        "scratch_path": "/shared/scratch"
    }'

The response holds one entry per detection (`rslp.vessels.VesselDetectionDict`), with
`source: "nisar"`, the position in both pixel and lon/lat coordinates, the detector
score, and, when `crop_path` is set, `crop_fnames` keyed `hh` and `hv`.

A granule is the only way to give the service imagery, since it has no data source of
its own to look one up with. Detections are labeled with `scene_id`, which the request
may set and which otherwise falls back to the granule filename.

The same pipeline is available as a workflow:

    python -m rslp.main nisar_vessels predict --tasks '[{"h5_path": "...", "json_path": "..."}]' --score_threshold 0.7

### How a granule becomes a window

GDAL cannot georeference the HDF5 datasets inside a granule, so `hdf5.py` reads the grid
with h5py (the `xCoordinates`/`yCoordinates`/`projection` datasets that sit alongside the
bands) and writes the HHHH and HVHV bands out as one GeoTIFF. That GeoTIFF is then an
ordinary `LocalFiles` raster layer, configured by `data/nisar_vessels/config_predict.json`.

The scene is then split into `NISAR_SCENE_TILE_SIZE` tiles, one rslearn window each,
rather than materialized as a single window covering the granule. rslearn builds a
window's raster as one in-memory array, so a whole-granule window makes peak memory
scale with the granule, and granule area varies widely across bandwidth modes. Tiling
caps it at the tile regardless of scene size. Tiles overlap so a vessel on a seam falls
fully inside one of them, and detections in the overlap band are deduplicated
afterwards.

Two further details are load-bearing, both so inference sees what training saw:

- The window is created in the UTM/UPS zone of the scene centroid at 10 m/pixel, the
  same way `create_dataset` builds training windows. GCOV is already geocoded, but not
  necessarily in that zone, and is posted at 10 m or 20 m.
- `config_predict.json` sets no `resampling_method`, so the layer inherits rslearn's
  bilinear default, which is what materialized the training dataset.

Detection crops are read straight back out of the scene GeoTIFF rather than materialized
into windows of their own, or cut from the tile the detection was found in. That read is
windowed, so it costs the crop rather than the scene, and a detection next to a tile seam
still gets a full crop instead of one half filled with nodata.

### Configuration

Every knob is named in `rslp/nisar_vessels/config.py`, which reads all of them from the
environment except `MARINE_INFRA_PATH` (read by `rslp.utils.filter` and re-exported here
so the service's settings stay in one place):

| Variable | Default | Purpose |
| --- | --- | --- |
| `NISAR_HOST` / `NISAR_PORT` | `0.0.0.0` / `5555` | Where the server binds. |
| `NISAR_SCORE_THRESHOLD` | `0.7` | Detector threshold, overridable per request. |
| `NISAR_INFRA_DISTANCE_KM` | `0.05` | Radius for dropping detections on marine infrastructure. |
| `MARINE_INFRA_PATH` | Satlas marine GeoJSON URL | The marine infrastructure to filter against. |
| `RSLEARN_NUM_DATA_LOADER_WORKERS` | `4` | Data loader workers during prediction. |
| `NISAR_PREPARE_WORKERS` | `32` | Workers used to prepare and ingest. |
| `NISAR_MATERIALIZE_WORKERS` | `8` | Workers used to materialize. |
| `NISAR_SCENE_TILE_SIZE` | `2048` | Tile the scene is split into for materialization. |
| `NISAR_SCENE_TILE_OVERLAP` | `64` | Overlap between adjacent scene tiles. |
| `NISAR_PREDICT_CROP_SIZE` | `128` | Tile size the detector runs over at inference. |
| `NISAR_PREDICT_OVERLAP_PIXELS` | `16` | Overlap between adjacent tiles. |

Peak memory during materialization is `NISAR_SCENE_TILE_SIZE` squared times
`NISAR_MATERIALIZE_WORKERS`, since each worker is a process holding a whole tile. Raising
either means lowering the other, and the pod's memory limit should be set against their
product rather than the tile alone.

The crop defaults match what the detector trained on. A larger crop means fewer forward
passes but not less compute, since the overlap fraction stays the same, so raise
`NISAR_PREDICT_CROP_SIZE` only if profiling shows per-crop overhead matters, and compare
detections against the default first.

### Building the image

    docker compose -f rslp/nisar_vessels/docker-compose.yaml build

CI publishes the image to GHCR as `allenai/nisar-vessel-detection` when a
`nisar_vessels_v*` tag is pushed (see `.github/workflows/nisar_vessel.yaml`), matching
how the other vessel services are released.

The Dockerfile downloads the detector checkpoint to the path implied by
`project_name`/`run_name` in `DETECT_MODEL_CONFIG`, so the two move together: pointing
the service at a different run means updating the config and the Dockerfile path.
That includes retraining, since each new run changes `run_name` in the config.
The deployed weights are the `data_20261014_satlas_00` run (`config_satlas.yaml`). On
the dataset_20261014 val split (89 vessels) it reaches 0.93 recall at 0.77 precision at
the default 0.7 threshold (AP 0.937). Recall is lower on 40 MHz granules (0.85 at 0.67
precision) than on 20 MHz ones (0.98 at 0.83).

The checkpoint is the only weights file the service loads. The Swin backbone is built
with `pretrained: false`, and the SatlasPretrain `restore_config` (a WEKA path) is only
applied when training starts, so the container downloads nothing at startup. The
Dockerfile sets `HF_HUB_OFFLINE=1` so an unexpected Hugging Face download fails instead.
Switching to a config whose encoder loads pretrained weights when it is built (OlmoEarth
by `model_id`, or Swin with `pretrained: true`) means baking those weights into the image
too, as the other vessel service Dockerfiles do.
