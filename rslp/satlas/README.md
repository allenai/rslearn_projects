This contains training, inference, and post-processing pipelines for the models served
at https://satlas.allen.ai/.

## Marine Infrastructure

Training:

    python -m rslp.rslearn_main model fit --config data/satlas/marine_infra/config.yaml

Inference:

    python -m rslp.main satlas write_jobs_for_year_months '[[2024, 7]]' MARINE_INFRA 'gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/{year:04d}-{month:02d}/' skylight-proto-1 rslp-job-queue-favyen

Post-processing:

    python -m rslp.main satlas merge_points MARINE_INFRA 2024-07 gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/2024-07/ gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/merged/
    python -m rslp.main satlas smooth_points MARINE_INFRA 2024-07 gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/merged/ gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/smoothed/
    python -m rslp.main satlas publish_points MARINE_INFRA gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/smoothed/ 'marine-default-cluster@v4'

## Wind Turbine

Training:

    python -m rslp.rslearn_main model fit --config data/satlas/wind_turbine/config.yaml

Inference:

    python -m rslp.main satlas write_jobs_for_year_months '[[2024, 1]]' WIND_TURBINE 'gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/{year:04d}-{month:02d}/' skylight-proto-1 rslp-job-queue-favyen --days_before 90 --days_after 181

Post-processing:

    python -m rslp.main satlas merge_points WIND_TURBINE 2024-01 gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/2024-01/ gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/merged/
    python -m rslp.main satlas smooth_points WIND_TURBINE 2024-01 gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/merged/ gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/smoothed/

Publishing for wind turbine is not supported yet since it needs to be combined with the
detected solar farms and published as "renewable energy" GeoJSON.
