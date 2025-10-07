"""Launch OlmoEarth fine-tuning evals."""

import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch OlmoEarth fine-tuning evaluations"
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        required=True,
        help="Models to evaluate",
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="*",
        required=True,
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Experiment prefix",
    )
    args = parser.parse_args()

    for model in args.model:
        for task in args.task:
            basic_args = [
                "python",
                "-m",
                "rslp.main",
                "common",
                "beaker_train",
                "--project_id",
                "2025_10_03_downstream_finetuning",
                "--experiment_id",
                f"{args.prefix}_{task}_{model}",
                "--cluster+=ai2/jupiter",
                "--cluster+=ai2/ceres",
                '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}',
                "--image_name",
                "favyen/rslphelios15",
                "--priority",
                "urgent",
                "--extra_env_vars",
                '{"EVAL_ADAPTER_MODEL_ID": "' + model + '"}',
            ]
            task_config_args = [
                f"--config_paths+=data/helios_v3/tasks/{cfg_fname}.yaml"
                for cfg_fname in TASK_CONFIGS[task]
            ]
            model_config_args = [f"--config_paths+=data/helios_v3/models/{model}.yaml"]
            subprocess.check_call(basic_args + task_config_args + model_config_args)  # nosec
