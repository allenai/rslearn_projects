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

Run from the rslearn_projects root:

    export STUDIO_API_KEY=...
    python -m rslp.nisar_vessels.scripts.create_dataset \
        --project_id c927e5cb-734b-4323-8cad-7f224b3e850d \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/dataset_20260828/

## 2. Prepare and materialize

This looks up matching NISAR L2 GCOV granules and writes the HHHH/HVHV rasters for
each window. It requires:

- olmoearth_run installed with the NISAR data source (currently the
  `patrickj/nisar-rslearn-data-source` branch), since the dataset config references
  `olmoearth_run...nisar.l2_gcov.NisarL2Gcov`.
- NASA Earthdata credentials in `EARTHDATA_CREDENTIALS` (JSON with `username` and
  `password`) to download the granules from ASF.

The NISAR layer is configured with `ingest: false`, so there is no separate
`rslearn dataset ingest` step; materialize reads directly from the granules.

    export EARTHDATA_CREDENTIALS='{"username": "...", "password": "..."}'
    rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/dataset_20260828/ --workers 32
    rslearn dataset materialize --root /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/dataset_20260828/ --workers 32

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
        --data.init_args.path /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/dataset_20260828/

## 4. Prediction windows (e.g. for mining hard negatives)

`create_predict_windows` searches the public ASF API (no credentials needed) for
NISAR L2 GCOV scenes acquired in a time range, keeps the dual-pol H-transmit ones
(the only mode with the HHHH/HVHV bands), and creates one unlabeled window per scene
at a random location within the scene footprint (deterministic per scene, so
re-running does not move existing windows). The windows go into the `predict` group
by default (matching `predict_config` in the model configs); use `--group` to
override.

Note the scene volume: globally there are roughly 1500 dual-pol scenes per day, so
use `--max_scenes` to randomly subsample. For example, 1000 windows of 2048x2048
over a two-month period:

    python -m rslp.nisar_vessels.scripts.create_predict_windows \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/nisar_vessels/dataset_20260828/ \
        --start_time 2026-06-01T00:00:00Z \
        --end_time 2026-08-01T00:00:00Z \
        --window_size 2048 \
        --max_scenes 1000

Then prepare/materialize just that group, and run prediction (optionally lowering
the confidence threshold to favor recall, so false positives can be reviewed and
added back to the dataset as negative examples):

    rslearn dataset prepare --root <ds_path> --group predict --workers 32
    rslearn dataset materialize --root <ds_path> --group predict --workers 32
    rslearn model predict \
        --config data/nisar_vessels/config_olmoearth_tiny.yaml \
        --data.init_args.path <ds_path> \
        --data.init_args.task.init_args.tasks.detect.init_args.score_threshold=0.1

Predictions are written to the `output` layer of each window.
