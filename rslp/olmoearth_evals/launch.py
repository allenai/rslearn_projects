"""Launch OlmoEarth fine-tuning evals."""

import json
import subprocess  # nosec

from rslearn.utils.geometry import PixelBounds

TASK_CONFIGS = {
    "africa_crop_mask": ["africa_crop_mask"],
    "awf_aef": ["awf_aef"],
    "awf_ts": ["awf_base"],
    "awf_mm": ["awf_base", "awf_mm"],
    "canada_crops_coarse": ["canada_crops_coarse"],
    "canada_crops_fine": ["canada_crops_fine"],
    "descals": ["descals"],
    "ecosystem_aef": ["ecosystem_aef"],
    "ethiopia_crops": ["ethiopia_crops"],
    "ecosystem": ["ecosystem"],
    "forest_loss_driver": ["forest_loss_driver"],
    "glance": ["glance"],
    "landsat_vessels": ["landsat_vessels"],
    "lcmap_lu": ["lcmap_lu"],
    "lfmc_aef": ["lfmc_aef"],
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
    "nandi_aef": ["nandi_aef"],
    "pastis_uni": ["pastis_base"],
    "pastis_ts": ["pastis_base", "pastis_ts"],
    "pastis_mm": ["pastis_base", "pastis_mm"],
    "sentinel1_vessels": ["sentinel1_vessels"],
    "sentinel2_vessel_length": ["sentinel2_vessel_length"],
    "sentinel2_vessel_type": ["sentinel2_vessel_type"],
    "sentinel2_vessels": ["sentinel2_vessels"],
    "solar_farm_aef": ["solar_farm_aef"],
    "solar_farm_uni": ["solar_farm_base"],
    "solar_farm_ts": ["solar_farm_base", "solar_farm_ts"],
    "solar_farm_mm": ["solar_farm_base", "solar_farm_mm"],
    "us_trees": ["us_trees"],
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
    clusters: list[str] = ["ai2/jupiter", "ai2/ceres"],
    test: bool = False,
    pad_to: int | None = None,
    crop_to: PixelBounds | None = None,
    use_embeddings: bool = False,
    model_config: dict[str, str] | None = None,
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
        test: whether to test the model instead of fit.
        pad_to: pad (or crop) input images to this size using center mode.
        crop_to: crop input images to these pixel bounds (col1, row1, col2, row2).
        use_embeddings: add an EmbeddingCache wrapping the encoder. This will ensure
            the encoder is not re-applied on crops that it has already processed, and
            also the encoder will not see gradients.
        model_config: optional dict of model configuration overrides. For example,
            {"decoder": "singleconv"} to use a single conv decoder for segmentation.
    """
    for model in models:
        for task in tasks:
            # Build env vars dict (pad_to and crop_to are passed as env vars
            # since they're on a transform inside a YAML list and can't be
            # easily overridden via jsonargparse CLI args).
            env_vars_dict: dict[str, str] = {"EVAL_ADAPTER_MODEL_ID": model}
            if pad_to is not None:
                env_vars_dict["EVAL_ADAPTER_PAD_TO"] = str(pad_to)
            if crop_to is not None:
                env_vars_dict["EVAL_ADAPTER_CROP_TO"] = ",".join(
                    str(x) for x in crop_to
                )
            if model_config is not None:
                env_vars_dict["EVAL_ADAPTER_MODEL_CONFIG"] = json.dumps(model_config)

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
                json.dumps(env_vars_dict),
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

            # Build extra_args for the training script.
            all_extra_args: list[str] = []
            if use_embeddings:
                all_extra_args.append(
                    "--model.init_args.model.init_args.use_embeddings=true"
                )
            if test:
                basic_args.extend(["--mode", "test"])
                all_extra_args.extend(["--log_mode=yes", "--load_checkpoint_mode=best"])
            if all_extra_args:
                basic_args.extend(["--extra_args", json.dumps(all_extra_args)])

            subprocess.check_call(
                basic_args + cluster_args + task_config_args + model_config_args
            )  # nosec
