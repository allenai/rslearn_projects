## Marine Infrastructure

Inference:

    python -m rslp.main satlas write_jobs_for_year_months '[[2024, 7]]' MARINE_INFRA 'gs://rslearn-eai/projects/satlas/marine_infra/version-20241212/{year:04d}-{month:02d}/' skylight-proto-1 rslp-job-queue-favyen

Post-processing:

## Wind Turbine

Inference:

    python -m rslp.main satlas write_jobs_for_year_months '[[2024, 1]]' WIND_TURBINE 'gs://rslearn-eai/projects/satlas/wind_turbine/version-20241210/{year:04d}-{month:02d}/' skylight-proto-1 rslp-job-queue-favyen --days_before 90 --days_after 181

Post-processing:
