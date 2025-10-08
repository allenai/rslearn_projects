"""Launch OlmoEarth fine-tuning evals."""

import subprocess  # nosec

TASK_CONFIGS = {
    "pastis_uni": ["pastis_base"],
    "pastis_ts": ["pastis_base", "pastis_ts"],
    "pastis_mm": ["pastis_base", "pastis_mm"],
    "marine_infra_uni": ["marine_infra_base"],
    "marine_infra_ts": ["marine_infra_base", "marine_infra_ts"],
    "marine_infra_mm": ["marine_infra_base", "marine_infra_mm"],
    "solar_farm_uni": ["solar_farm_base"],
    "solar_farm_ts": ["solar_farm_base", "solar_farm_ts"],
    "solar_farm_mm": ["solar_farm_base", "solar_farm_mm"],
    "wind_turbine_uni": ["wind_turbine_base"],
    "wind_turbine_ts": ["wind_turbine_base", "wind_turbine_ts"],
    "wind_turbine_mm": ["wind_turbine_base", "wind_turbine_mm"],
    "sentinel1_vessels": ["sentinel1_vessels"],
    "sentinel2_vessel_length": ["sentinel2_vessel_length"],
    "sentinel2_vessel_type": ["sentinel2_vessel_type"],
    "sentinel2_vessels": ["sentinel2_vessels"],
}


def launch(
    models: list[str],
    tasks: list[str],
    prefix: str,
    image_name: str,
    priority: str = "high",
) -> None:
    """Launch OlmoEarth fine-tuning evaluation.

    Args:
        models: the models to run. For example, ["olmoearth", "satlaspretrain"]. See
            data/olmoearth_evals/models/ for available configs.
        tasks: the tasks to run. See data/olmoearth_evals/tasks/ for available configs.
        prefix: prefix for this experiment.
        image_name: the Beaker image name to use.
        priority: the Beaker priority to use.
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
                "2025_10_03_downstream_finetuning",
                "--experiment_id",
                f"{prefix}_{task}_{model}",
                "--cluster+=ai2/jupiter",
                "--cluster+=ai2/ceres",
                '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}',
                "--image_name",
                image_name,
                "--priority",
                priority,
                "--extra_env_vars",
                '{"EVAL_ADAPTER_MODEL_ID": "' + model + '"}',
            ]
            task_config_args = [
                f"--config_paths+=data/olmoearth_evals/tasks/{cfg_fname}.yaml"
                for cfg_fname in TASK_CONFIGS[task]
            ]
            model_config_args = [
                f"--config_paths+=data/olmoearth_evals/models/{model}.yaml"
            ]
            subprocess.check_call(basic_args + task_config_args + model_config_args)  # nosec
