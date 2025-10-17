"""Launch OlmoEarth fine-tuning evals."""

import subprocess  # nosec

TASK_CONFIGS = {
    "awf_ts": ["awf_base"],
    "awf_mm": ["awf_base", "awf_mm"],
    "ecosystem": ["ecosystem"],
    "forest_loss_driver": ["forest_loss_driver"],
    "landsat_vessels": ["landsat_vessels"],
    "lfmc_uni": ["lfmc_base"],
    "lfmc_ts": ["lfmc_base", "lfmc_ts"],
    "lfmc_mm": ["lfmc_base", "lfmc_mm"],
    "mangrove_uni": ["mangrove_base"],
    "mangrove_ts": ["mangrove_base", "mangrove_ts"],
    "mangrove_mm": ["mangrove_base", "mangrove_mm"],
    "marine_infra_uni": ["marine_infra_base"],
    "marine_infra_ts": ["marine_infra_base", "marine_infra_ts"],
    "marine_infra_mm": ["marine_infra_base", "marine_infra_mm"],
    "nandi_ts": ["nandi_base"],
    "nandi_mm": ["nandi_base", "nandi_mm"],
    "pastis_uni": ["pastis_base"],
    "pastis_ts": ["pastis_base", "pastis_ts"],
    "pastis_mm": ["pastis_base", "pastis_mm"],
    "sentinel1_vessels": ["sentinel1_vessels"],
    "sentinel2_vessel_length": ["sentinel2_vessel_length"],
    "sentinel2_vessel_type": ["sentinel2_vessel_type"],
    "sentinel2_vessels": ["sentinel2_vessels"],
    "solar_farm_uni": ["solar_farm_base"],
    "solar_farm_ts": ["solar_farm_base", "solar_farm_ts"],
    "solar_farm_mm": ["solar_farm_base", "solar_farm_mm"],
    "wind_turbine_uni": ["wind_turbine_base"],
    "wind_turbine_ts": ["wind_turbine_base", "wind_turbine_ts"],
    "wind_turbine_mm": ["wind_turbine_base", "wind_turbine_mm"],
}


def launch(
    models: list[str],
    tasks: list[str],
    prefix: str,
    image_name: str,
    project: str,
    priority: str = "high",
    clusters: list[str] = ["ai2/jupiter", "ai2/ceres", "ai2/titan"],
) -> None:
    """Launch OlmoEarth fine-tuning evaluation.

    Args:
        models: the models to run. For example, ["olmoearth", "satlaspretrain"]. See
            data/olmoearth_evals/models/ for available configs.
        tasks: the tasks to run. See data/olmoearth_evals/tasks/ for available configs.
        prefix: prefix for this experiment.
        image_name: the Beaker image name to use.
        project: W&B project name.
        priority: the Beaker priority to use.
        clusters: Beaker clusters to target.
    """
    for model in models:
        for task in tasks:
            basic_args = [
                "python",
                "-m",
                "rslp.main",
                "common",
                "beaker_train",
                "--project_id",
                project,
                "--experiment_id",
                f"{prefix}_{task}_{model}",
                '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}',
                "--image_name",
                image_name,
                "--priority",
                priority,
                "--extra_env_vars",
                '{"EVAL_ADAPTER_MODEL_ID": "' + model + '"}',
            ]

            cluster_args = [f"--cluster+={cluster}" for cluster in clusters]

            task_config_args = [
                f"--config_paths+=data/olmoearth_evals/tasks/{cfg_fname}.yaml"
                for cfg_fname in TASK_CONFIGS[task]
            ]

            if task == "forest_loss_driver":
                # Need to use different config for it to work properly because the
                # model architecture is different.
                if model == "satlaspretrain":
                    # For SatlasPretrain the config has not just freezing but also
                    # restoring model.
                    model_config_fname = "forest_loss_driver_satlaspretrain.yaml"
                else:
                    # Otherwise we can always freeze the same portion I think.
                    model_config_fname = "forest_loss_driver.yaml"
                model_config_args = [
                    f"--config_paths+=data/olmoearth_evals/models/{model_config_fname}"
                ]
            else:
                model_config_args = [
                    f"--config_paths+=data/olmoearth_evals/models/{model}.yaml"
                ]

            subprocess.check_call(
                basic_args + cluster_args + task_config_args + model_config_args
            )  # nosec
