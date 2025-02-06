This contains training, inference, and post-processing pipelines for the models served
at https://satlas.allen.ai/.

## Marine Infrastructure

The marine infrastructure model detects off-shore infrastructure in two categories:
off-shore wind turbines and off-shore platforms. The latter category essentially
includes any manmade object in the ocean that is stationary and would not normally be
considered an artificial island.

The model inputs four Sentinel-2 images, where each image should be a mosaic that uses
Sentinel-2 scenes from a distinct month. The dataset configuration uses
`MonthlySentinel2` in `rslp/satlas/data_sources.py` to achieve this, and only uses
Sentinel-2 scenes with at most 50% cloud cover. If a given month does not have enough
matching images under the cloud threshold, then images from earlier months may be used.

The model is meant to be run on a quarterly basis, using images up to 4 months
before the start of the quarter (giving 7 possible months to pick 4 mosaics from). If
all of the months are cloudy, then it is okay to skip that inference and come back to
it in a later season that may be less cloudy. At the same time, we don't want to limit
to just 4 months because a region may never have 4 consecutive months with cloud-free
images available.

### Training

The model is trained using the rslearn dataset in `gs://rslearn-eai`. See the model
configuration file for more details.

    python -m rslp.rslearn_main model fit --config data/satlas/marine_infra/config.yaml

### Inference

For inference, the world is split up into ~10K tiles in each of the 60 UTM zones,
yielding 600K inference tasks in total. For each task, an inference worker will:

1. Create an rslearn dataset with a single window corresponding to the UTM zone/tile.
2. Execute the data ingestion pipeline to populate that window with Sentinel-2 images.
3. Apply the model on the window to create an output layer in the rslearn dataset.
4. Copy the contents of that output layer to a location on GCS.

The task queue system is implemented using `rslp.common.worker`, see
`rslp/common/README.md` for details. Essentially, we first write tasks to a Google
Cloud Pub/Sub topic, and then launch workers that will read from the topic.

Then, we start by writing the tasks:

    python -m rslp.main satlas write_jobs_for_year_months '[[2024, 7]]' MARINE_INFRA 'gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/{year:04d}-{month:02d}/' skylight-proto-1 rslp-job-queue-favyen --days_before 120 --days_after 90

Here:

- `[[2024, 7]]` is a list of year-month pairs that we want to run the model on.
- MARINE_INFRA is the application we want to apply. This is an enum for "marine_infra"
  and it will automatically use the dataset configuration at
  `data/satlas/marine_infra/config.json` and the model configuration at
  `data/satlas/marine_infra/config.yaml`.
- `gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/{year:04d}-{month:02d}/`
  is the path where outputs should be written. Outputs will be named like
  `EPSG:32601_65536_-524288.geojson` where `EPSG:32601` is the UTM projection, and
  65536 and -524288 are the starting column and row (respectively) of the tile. The
  path should have a year and month placeholder.
- `skylight-proto-1` is the project of the Pub/Sub topic.
- `rslp-job-queue-favyen` is the name of the Pub/Sub topic.
- The inference tasks should create a window spanning 120 days before the specified
  timestamp (to use images before the quarter when necessary) and 90 days after the
  timestamp (corresponding to the duration of the quarter).

Then start the workers. See `rslp/common/README.md` for details. In this example,
`rslp-job-queue-favyen-sub` should be a subscription for the topic to which the tasks
were written. Here we start 100 workers.

    python -m rslp.main common launch skylight-proto-1 rslp-job-queue-favyen-sub 100 --gpus 1 --shared_memory 256GiB

### Post-processing.

Post-processing for point tasks occurs locally (does not require starting jobs in parallel).

First, merge the points computed across all of the different tasks:

    python -m rslp.main satlas merge_points MARINE_INFRA 2024-07 gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/2024-07/ gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/merged/

Here:

- MARINE_INFRA is the application we want to apply.
- 2024-07 is the timestep label. All timestep labels are YYYY-MM for the Satlas
  systems.
- `gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/2024-07/` is the
  folder containing inference outputs that we want to merge.
- `gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/merged/` is the
  folder to write merged outputs. The output filename will be
  `gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/merged/2024-07.geojson`.

Second, smooth the points across timesteps. This runs a Viterbi smoothing operation.
Note that the Viterbi smoothing is implemented in a separate Go application at
`rslp/satlas/scripts/smooth_point_labels_viterbi.go`.

    python -m rslp.main satlas smooth_points MARINE_INFRA 2024-07 gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/merged/ gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/smoothed/

Finally, publish the outputs to Cloudflare R2.

    python -m rslp.main satlas publish_points MARINE_INFRA gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/smoothed/ 'marine-default-cluster@v4'

## Wind Turbine

The wind turbine model detects wind turbines on land. It is meant to be run on a
semi-annual basis, and inputs six Sentinel-2 images. As with the marine infrastructure
model, each is a mosaic using images within a 30-day period.

Training:

    python -m rslp.rslearn_main model fit --config data/satlas/wind_turbine/config.yaml

Inference:

    python -m rslp.main satlas write_jobs_for_year_months '[[2024, 1]]' WIND_TURBINE 'gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/{year:04d}-{month:02d}/' skylight-proto-1 rslp-job-queue-favyen --days_before 90 --days_after 181

Post-processing:

    python -m rslp.main satlas merge_points WIND_TURBINE 2024-01 gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/2024-01/ gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/merged/
    python -m rslp.main satlas smooth_points WIND_TURBINE 2024-01 gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/merged/ gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/smoothed/

Publishing for wind turbine is not supported yet since it needs to be combined with the
detected solar farms and published as "renewable energy" GeoJSON.
