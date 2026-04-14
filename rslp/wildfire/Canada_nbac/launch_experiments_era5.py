"""Launch multiple CanadaFireSat experiments with programmatic config generation.

Each experiment is defined as a dict of overrides applied to the base config.
This avoids any issues with file-save races on Weka or stale editor buffers,
because the config is generated fresh in memory and written to a unique file
right before each launch.

Usage:
    cd /weka/dfive-default/hadriens/rslearn_projects
    source ./.venv/bin/activate
    python -m rslp.wildfire.Canada_nbac.launch_experiments_era5

    # Or run only specific experiments by name:
    python -m rslp.wildfire.Canada_nbac.launch_experiments_era5 --only exp1_lr1e3 exp3_heavy_dropout

    # Dry-run mode (generates configs and prints commands without launching):
    python -m rslp.wildfire.Canada_nbac.launch_experiments_era5 --dry-run

    # Local mode (runs rslearn model fit locally with test dataset, no wandb):
    python -m rslp.wildfire.Canada_nbac.launch_experiments_era5 --local
    python -m rslp.wildfire.Canada_nbac.launch_experiments_era5 --local --only exp1
    python -m rslp.wildfire.Canada_nbac.launch_experiments_era5 --local --dry-run
"""

import argparse
import copy
import os
import shutil
import subprocess  # nosec B404
import sys
from datetime import datetime
from typing import Any, cast

import yaml

# ERA5 per-channel normalization stats (from era5_365dhistory_norm_stats.json)
# Band order: d2m, e, pev, ro, sp, ssr, ssrd, str, swvl1, swvl2, t2m, tp, u10, v10
ERA5_MEAN = [
    -4.53467423,
    -0.00112198,
    -0.00834321,
    0.00091739,
    94479.34979023,
    9075594.89242382,
    11654781.72080826,
    -4500463.10594404,
    0.34399074,
    0.34002065,
    0.16157731,
    0.00190445,
    0.39923426,
    0.10049819,
]
ERA5_STD = [
    12.51564845,
    0.00121721,
    0.00957517,
    0.00220943,
    4852.98599929,
    7488882.82982511,
    8471632.012075,
    2226365.36681933,
    0.13658633,
    0.13478928,
    13.64278631,
    0.00426817,
    1.77886309,
    1.67962054,
]

# ERA5 8-day forecast normalization stats (from era5_8dforecast_norm_stats.json)
ERA5_FORECAST_MEAN = [
    -3.30546649,
    -0.00115266,
    -0.01093713,
    0.00134583,
    94582.0162208,
    12499732.41511201,
    17005837.31171564,
    -5072686.89569636,
    0.36648215,
    0.36040571,
    2.29299751,
    0.00164887,
    0.08920109,
    0.07433319,
]
ERA5_FORECAST_STD = [
    9.1623422,
    0.00105223,
    0.00964415,
    0.00316157,
    4863.30295829,
    6777222.05066305,
    7399997.35708814,
    2194645.74649882,
    0.13852829,
    0.13592012,
    10.18087003,
    0.00370383,
    1.81719633,
    1.71739672,
]

# ──────────────────────────────────────────────────────────────────────────────
# AUGMENTATION TRANSFORMS (training-only, applied after normalization)
# ──────────────────────────────────────────────────────────────────────────────
_MASK_TRANSFORM = {
    "class_path": "rslearn.train.transforms.ts_augment.RandomTimeMasking",
    "init_args": {
        "mask_ratio": 0.10,
        "selectors": ["era5_daily"],
    },
}
_NOISE_TRANSFORM = {
    "class_path": "rslearn.train.transforms.gaussian_noise.GaussianNoise",
    "init_args": {
        "std": 0.02,
        "selectors": ["era5_daily"],
    },
}
_SHIFT_TRANSFORM = {
    "class_path": "rslearn.train.transforms.ts_augment.TemporalShift",
    "init_args": {
        "max_shift": 2,
        "selectors": ["era5_daily"],
    },
}


def _train_transforms(*augments: dict) -> list[dict]:
    """Build train_config transforms: normalize → [augments] → maxpool."""
    pre = [
        {
            "class_path": "rslearn.train.transforms.normalize.Normalize",
            "init_args": {
                "mean": ERA5_MEAN,
                "std": ERA5_STD,
                "selectors": ["era5_daily"],
            },
        },
    ]
    post = [
        {
            "class_path": "rslearn.train.transforms.resize.MaxPoolResize",
            "init_args": {
                "target_size": [1, 1],
                "selectors": [
                    "target/segmentation/classes",
                    "target/segmentation/valid",
                ],
            },
        },
    ]
    return pre + list(augments) + post


def _eval_transforms(
    *extra_transforms: dict, include_target_resize: bool = True
) -> list[dict]:
    """Build non-train transforms with optional extra ERA5 transforms."""
    transforms = [
        {
            "class_path": "rslearn.train.transforms.normalize.Normalize",
            "init_args": {
                "mean": ERA5_MEAN,
                "std": ERA5_STD,
                "selectors": ["era5_daily"],
            },
        },
        *list(extra_transforms),
    ]
    if include_target_resize:
        transforms.append(
            {
                "class_path": "rslearn.train.transforms.resize.MaxPoolResize",
                "init_args": {
                    "target_size": [1, 1],
                    "selectors": [
                        "target/segmentation/classes",
                        "target/segmentation/valid",
                    ],
                },
            }
        )
    return transforms


# ──────────────────────────────────────────────────────────────────────────────
# BASE CONFIG — this is your "template". Edit it here, not in the YAML file.
# ──────────────────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "model": {
        "class_path": "rslearn.train.lightning_module.RslearnLightningModule",
        "init_args": {
            "model": {
                "class_path": "rslearn.models.multitask.MultiTaskModel",
                "init_args": {
                    "encoder": [
                        {
                            "class_path": "rslearn.models.tcn_encoder.TCNEncoder",  # TCNEncoder, SimpleTCNEncoder
                            "init_args": {
                                "in_channels": 14,
                                "d_model": 128,
                                # "base_dim": 32,
                                "d_output": 2,
                                # "output_dim": 2,
                                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                                "pooling_windows": [1, 2, 4, 12],
                                "dropout": 0.1,
                                "output_spatial_size": 1,
                            },
                        }
                    ],
                    "decoders": {
                        "segmentation": [
                            {
                                "class_path": "rslearn.train.tasks.segmentation.SegmentationHead",
                                "init_args": {
                                    "weights": [0.4, 0.6],
                                },
                            }
                        ]
                    },
                },
            },
            "optimizer": {
                "class_path": "rslearn.train.optimizer.AdamW",
                "init_args": {
                    "lr": 0.001,
                    "weight_decay": 0.01,
                },
            },
            "scheduler": {
                "class_path": "rslearn.train.scheduler.CosineAnnealingScheduler",
                "init_args": {
                    "T_max": 40,
                    "eta_min_factor": 0.01,
                },
            },
        },
    },
    "data": {
        "class_path": "rslearn.train.data_module.RslearnDataModule",
        "init_args": {
            "path": "/weka/dfive-default/rslearn-eai/datasets/wildfire/canada_nbac",
            "inputs": {
                "era5_daily": {
                    "data_type": "raster",
                    "layers": ["era5_365dhistory"],
                    "use_all_bands_in_order_of_band_set_idx": 0,
                    "passthrough": True,
                    "dtype": "FLOAT32",
                },
                "label": {
                    "data_type": "raster",
                    "layers": ["label_100m"],
                    "bands": ["label_100m"],
                    "dtype": "INT32",
                    "is_target": True,
                },
            },
            "task": {
                "class_path": "rslearn.train.tasks.multi_task.MultiTask",
                "init_args": {
                    "tasks": {
                        "segmentation": {
                            "class_path": "rslearn.train.tasks.segmentation.SegmentationTask",
                            "init_args": {
                                "num_classes": 2,
                                "enable_miou_metric": True,
                                "enable_f1_metric": True,
                                "report_metric_per_class": True,
                                "other_metrics": {
                                    "PRAUC_cls0": {
                                        "class_path": "rslearn.train.tasks.segmentation.SegmentationMetric",
                                        "init_args": {
                                            "metric": {
                                                "class_path": "torchmetrics.classification.MulticlassAveragePrecision",
                                                "init_args": {
                                                    "num_classes": 2,
                                                    "average": None,
                                                    "thresholds": 128,
                                                },
                                            },
                                            "class_idx": 0,
                                        },
                                    },
                                    "PRAUC_cls1": {
                                        "class_path": "rslearn.train.tasks.segmentation.SegmentationMetric",
                                        "init_args": {
                                            "metric": {
                                                "class_path": "torchmetrics.classification.MulticlassAveragePrecision",
                                                "init_args": {
                                                    "num_classes": 2,
                                                    "average": None,
                                                    "thresholds": 128,
                                                },
                                            },
                                            "class_idx": 1,
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "input_mapping": {
                        "segmentation": {
                            "label": "targets",
                        },
                    },
                },
            },
            "batch_size": 64,
            "num_workers": 16,
            "default_config": {
                "transforms": [
                    {
                        "class_path": "rslearn.train.transforms.normalize.Normalize",
                        "init_args": {
                            "mean": ERA5_MEAN,
                            "std": ERA5_STD,
                            "selectors": ["era5_daily"],
                        },
                    },
                    {
                        "class_path": "rslearn.train.transforms.resize.MaxPoolResize",
                        "init_args": {
                            "target_size": [1, 1],
                            "selectors": [
                                "target/segmentation/classes",
                                "target/segmentation/valid",
                            ],
                        },
                    },
                ],
            },
            "train_config": {
                "groups": ["train"],
                "transforms": [
                    {
                        "class_path": "rslearn.train.transforms.normalize.Normalize",
                        "init_args": {
                            "mean": ERA5_MEAN,
                            "std": ERA5_STD,
                            "selectors": ["era5_daily"],
                        },
                    },
                    {
                        "class_path": "rslearn.train.transforms.resize.MaxPoolResize",
                        "init_args": {
                            "target_size": [1, 1],
                            "selectors": [
                                "target/segmentation/classes",
                                "target/segmentation/valid",
                            ],
                        },
                    },
                ],
            },
            "val_config": {
                "groups": ["val"],
            },
            "test_config": {
                "groups": ["test"],
            },
            "predict_config": {
                "skip_targets": True,
            },
        },
    },
    "trainer": {
        "max_epochs": 40,
        "logger": {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": {
                "group": "era5_hist",
            },
        },
        "callbacks": [
            {
                "class_path": "lightning.pytorch.callbacks.progress.tqdm_progress.TQDMProgressBar"
            },
            {
                "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                "init_args": {"logging_interval": "epoch"},
            },
            {
                "class_path": "rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint",
                "init_args": {
                    "monitor": "val_segmentation/PRAUC_cls1",
                    "mode": "max",
                },
            },
        ],
    },
    "management_dir": "${RSLP_PREFIX}/projects",
    "project_name": "20260321_wf_nbac_newsample",
    "run_name": "placeholder",  # overridden per experiment
}


# ──────────────────────────────────────────────────────────────────────────────
# LAUNCH SETTINGS — shared across all experiments
# ──────────────────────────────────────────────────────────────────────────────
LAUNCH_SETTINGS = {
    "image_name": "hadriens/rslpomp_hspec_260327_checkp",
    "cluster": "ai2/saturn",
    "gpus": 1,
    "weka_mounts": [
        {"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}
    ],
    "priority": "high",
}


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS — define each experiment as a (name, overrides_dict) pair.
# The overrides are applied on top of a deep copy of BASE_CONFIG.
# ──────────────────────────────────────────────────────────────────────────────
def set_nested(d: dict | list, key_path: str, value: Any) -> None:
    """Set a value in a nested dict/list using dot-separated key path.

    Numeric path segments are treated as list indices.
    Example: set_nested(cfg, "model.init_args.optimizer.init_args.lr", 0.0001)
    Example: set_nested(cfg, "encoder.0.init_args.dropout", 0.3)
    """
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d[int(k)] if isinstance(d, list) else d[k]
    last = keys[-1]
    if isinstance(d, list):
        d[int(last)] = value
    else:
        d[last] = value


def make_experiment(name: str, overrides: dict[str, Any]) -> tuple[str, dict]:
    """Create an experiment config by applying overrides to the base config.

    Args:
        name: experiment name (used as run_name and experiment_id).
        overrides: dict mapping dot-separated key paths to values.

    Returns:
        (name, config_dict) tuple
    """
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["run_name"] = name
    for key_path, value in overrides.items():
        set_nested(cfg, key_path, value)
    return (name, cfg)


def _transformer_encoder_init_args(**overrides: Any) -> dict[str, Any]:
    """Build a clean PatchTransformerEncoder init_args dict."""
    init_args = {
        "in_channels": 14,
        "d_model": 192,
        "d_output": 2,
        "num_layers": 4,
        "num_heads": 4,
        "patch_kernel_size": 14,
        "patch_stride": 7,
        "mlp_ratio": 4,
        "head_mlp_ratio": 2,
        "dropout": 0.2,
        "attention_dropout": 0.1,
        "drop_path_rate": 0.1,
        "pooling": "attention",
        "add_day_of_year_features": True,
        "add_relative_position_features": True,
        "position_encoding": "sinusoidal",
        "max_position_embeddings": 512,
        "mod_key": "era5_daily",
        "output_spatial_size": 1,
        "has_mask_channel": False,
        "pad_value": 0.0,
    }
    init_args.update(overrides)
    return init_args


# ──────────────────────────────────────────────────────────────────────────────
# FORECAST (8d) TRANSFORM HELPERS — use ERA5_FORECAST_{MEAN,STD}
# ──────────────────────────────────────────────────────────────────────────────
_FORECAST_MASK_TRANSFORM = {
    "class_path": "rslearn.train.transforms.ts_augment.RandomTimeMasking",
    "init_args": {
        "mask_ratio": 0.10,
        "selectors": ["era5_daily"],
    },
}
_FORECAST_NOISE_TRANSFORM = {
    "class_path": "rslearn.train.transforms.gaussian_noise.GaussianNoise",
    "init_args": {
        "std": 0.02,
        "selectors": ["era5_daily"],
    },
}
_FORECAST_SHIFT_TRANSFORM = {
    "class_path": "rslearn.train.transforms.ts_augment.TemporalShift",
    "init_args": {
        "max_shift": 2,
        "selectors": ["era5_daily"],
    },
}


def _forecast_train_transforms(*augments: dict) -> list[dict]:
    """Build train transforms for the 8d forecast layer: normalize → [augments] → maxpool."""
    pre = [
        {
            "class_path": "rslearn.train.transforms.normalize.Normalize",
            "init_args": {
                "mean": ERA5_FORECAST_MEAN,
                "std": ERA5_FORECAST_STD,
                "selectors": ["era5_daily"],
            },
        },
    ]
    post = [
        {
            "class_path": "rslearn.train.transforms.resize.MaxPoolResize",
            "init_args": {
                "target_size": [1, 1],
                "selectors": [
                    "target/segmentation/classes",
                    "target/segmentation/valid",
                ],
            },
        },
    ]
    return pre + list(augments) + post


def _forecast_eval_transforms() -> list[dict]:
    """Build eval transforms for the 8d forecast layer."""
    return [
        {
            "class_path": "rslearn.train.transforms.normalize.Normalize",
            "init_args": {
                "mean": ERA5_FORECAST_MEAN,
                "std": ERA5_FORECAST_STD,
                "selectors": ["era5_daily"],
            },
        },
        {
            "class_path": "rslearn.train.transforms.resize.MaxPoolResize",
            "init_args": {
                "target_size": [1, 1],
                "selectors": [
                    "target/segmentation/classes",
                    "target/segmentation/valid",
                ],
            },
        },
    ]


def _simple_tcn_init_args(**overrides: Any) -> dict[str, Any]:
    """Build SimpleTCNEncoder init_args for 8-timestep forecast data."""
    init_args = {
        "in_channels": 14,
        "num_conv_layers": 2,
        "base_dim": 32,
        "width_mult": 2,
        "mlp_ratio": 2,
        "output_dim": 2,
        "start_kernel_size": 5,
        "dropout": 0.1,
        "output_spatial_size": 1,
    }
    init_args.update(overrides)
    return init_args


def _tcn_init_args_8d(**overrides: Any) -> dict[str, Any]:
    """Build TCNEncoder init_args tuned for 8-timestep forecast data."""
    init_args = {
        "in_channels": 14,
        "d_model": 64,
        "d_output": 2,
        "kernel_size": 3,
        "dilations": [1, 2, 4],
        "pooling_windows": [1, 2, 4],
        "dropout": 0.1,
        "output_spatial_size": 1,
    }
    init_args.update(overrides)
    return init_args


_FORECAST_OVERRIDES_BASE: dict[str, Any] = {
    "data.init_args.inputs.era5_daily.layers": ["era5_8dforecast"],
    "data.init_args.default_config.transforms": _forecast_eval_transforms(),
    "data.init_args.train_config.transforms": _forecast_train_transforms(
        _FORECAST_MASK_TRANSFORM, _FORECAST_SHIFT_TRANSFORM, _FORECAST_NOISE_TRANSFORM
    ),
    "trainer.logger.init_args.group": "era5_8df",
}


_MASK_TRANSFORM_WITH_CHANNEL = {
    "class_path": "rslearn.train.transforms.ts_augment.RandomTimeMasking",
    "init_args": {
        "mask_ratio": 0.10,
        "selectors": ["era5_daily"],
        "append_mask_channel": True,
    },
}

_VALID_MASK_CHANNEL_TRANSFORM = {
    "class_path": "rslearn.train.transforms.ts_augment.RandomTimeMasking",
    "init_args": {
        "mask_ratio": 0.0,
        "selectors": ["era5_daily"],
        "append_mask_channel": True,
    },
}


# ──────────────────────────────── YOUR EXPERIMENTS ─────────────────────────────
# Edit this list to define your experiment variants.
# Each entry is: make_experiment("experiment_name", {dotted.key.path: value, ...})
#
# Example overrides you can use:
#   "model.init_args.optimizer.init_args.lr": 0.0003
#   "model.init_args.model.init_args.encoder.0.init_args.dropout": 0.3
#   "model.init_args.model.init_args.encoder.0.init_args.d_model": 256
#   "data.init_args.batch_size": 32
#   "trainer.max_epochs": 60
#   "model.init_args.model.init_args.decoders.segmentation.0.init_args.weights": [0.1, 0.9]

EXPERIMENTS = [
    # make_experiment("era5d_xfmr_base_14x7_attn_sin_doy_rel_lr1e4", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(),
    #     "model.init_args.optimizer.init_args.lr": 0.0001,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    # make_experiment("era5d_xfmr_base_14x7_attn_sin_doy_rel_lr5e5", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(),
    #     "model.init_args.optimizer.init_args.lr": 0.00005,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    # make_experiment("era5d_xfmr_base_14x7_attn_sin_doy_rel_lr1e5", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(),
    #     "model.init_args.optimizer.init_args.lr": 0.00001,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    # make_experiment("era5d_xfmr_base_14x7_attn_learned_doy_rel", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(
    #         position_encoding="learned",
    #     ),
    #     "model.init_args.optimizer.init_args.lr": 0.0001,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    # make_experiment("era5d_xfmr_base_14x7_attn_sin_no_doy_rel", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(
    #         add_day_of_year_features=False,
    #     ),
    #     "model.init_args.optimizer.init_args.lr": 0.0001,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    # make_experiment("era5d_xfmr_base_14x7_attn_sin_doy_norel", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(
    #         add_relative_position_features=False,
    #     ),
    #     "model.init_args.optimizer.init_args.lr": 0.0001,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    # make_experiment("era5d_xfmr_finepatch_7x7_attn_sin_doy_rel", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(
    #         patch_kernel_size=7,
    #         patch_stride=7,
    #     ),
    #     "model.init_args.optimizer.init_args.lr": 0.0001,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    # make_experiment("era5d_xfmr_coarsepatch_30x15_attn_sin_doy_rel", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(
    #         patch_kernel_size=30,
    #         patch_stride=15,
    #     ),
    #     "model.init_args.optimizer.init_args.lr": 0.0001,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    # make_experiment("era5d_xfmr_base_14x7_clsmean_sin_doy_rel", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(
    #         pooling="cls_mean_concat",
    #     ),
    #     "model.init_args.optimizer.init_args.lr": 0.0001,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    # make_experiment("era5d_xfmr_large_14x7_attn_sin_doy_rel", {
    #     "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
    #     "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(
    #         d_model=256,
    #         num_layers=6,
    #         num_heads=8,
    #         dropout=0.3,
    #         drop_path_rate=0.15,
    #     ),
    #     "model.init_args.optimizer.init_args.lr": 0.0001,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.03,
    #     "data.init_args.batch_size": 32,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
    # }),
    make_experiment(
        "era5d_xfmr_base_14x7_attn_sin_doy_rel_maskaware_aug",
        {
            "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
            "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(
                has_mask_channel=True,
            ),
            "model.init_args.optimizer.init_args.lr": 0.0001,
            "model.init_args.optimizer.init_args.weight_decay": 0.03,
            "data.init_args.train_config.transforms": _train_transforms(
                _MASK_TRANSFORM_WITH_CHANNEL, _SHIFT_TRANSFORM, _NOISE_TRANSFORM
            ),
            "data.init_args.val_config.transforms": _eval_transforms(
                _VALID_MASK_CHANNEL_TRANSFORM
            ),
            "data.init_args.test_config.transforms": _eval_transforms(
                _VALID_MASK_CHANNEL_TRANSFORM
            ),
            "data.init_args.predict_config.transforms": _eval_transforms(
                _VALID_MASK_CHANNEL_TRANSFORM
            ),
            "trainer.max_epochs": 60,
            "model.init_args.scheduler.init_args.T_max": 60,
            "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
        },
    ),
    make_experiment(
        "era5d_xfmr_base_14x7_attn_sin_doy_rel_maskaware_aug_w2575",
        {
            "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
            "model.init_args.model.init_args.encoder.0.init_args": _transformer_encoder_init_args(
                has_mask_channel=True,
            ),
            "model.init_args.optimizer.init_args.lr": 0.0001,
            "model.init_args.optimizer.init_args.weight_decay": 0.03,
            "model.init_args.model.init_args.decoders.segmentation.0.init_args.weights": [
                0.25,
                0.75,
            ],
            "data.init_args.train_config.transforms": _train_transforms(
                _MASK_TRANSFORM_WITH_CHANNEL, _SHIFT_TRANSFORM, _NOISE_TRANSFORM
            ),
            "data.init_args.val_config.transforms": _eval_transforms(
                _VALID_MASK_CHANNEL_TRANSFORM
            ),
            "data.init_args.test_config.transforms": _eval_transforms(
                _VALID_MASK_CHANNEL_TRANSFORM
            ),
            "data.init_args.predict_config.transforms": _eval_transforms(
                _VALID_MASK_CHANNEL_TRANSFORM
            ),
            "trainer.max_epochs": 60,
            "model.init_args.scheduler.init_args.T_max": 60,
            "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
        },
    ),
    # ──────────────────────── ERA5 8-day forecast — SimpleTCNEncoder ──────────────
    make_experiment(
        "era5_8df_simpleTCN_2L_d32_k5_lr1e3",
        {
            "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.tcn_encoder.SimpleTCNEncoder",
            "model.init_args.model.init_args.encoder.0.init_args": _simple_tcn_init_args(),
            "model.init_args.optimizer.init_args.lr": 0.001,
            "model.init_args.optimizer.init_args.weight_decay": 0.01,
            **_FORECAST_OVERRIDES_BASE,
        },
    ),
    make_experiment(
        "era5_8df_simpleTCN_3L_d32_k5_lr5e4",
        {
            "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.tcn_encoder.SimpleTCNEncoder",
            "model.init_args.model.init_args.encoder.0.init_args": _simple_tcn_init_args(
                num_conv_layers=3,
            ),
            "model.init_args.optimizer.init_args.lr": 0.0005,
            "model.init_args.optimizer.init_args.weight_decay": 0.01,
            **_FORECAST_OVERRIDES_BASE,
        },
    ),
    make_experiment(
        "era5_8df_simpleTCN_2L_d64_k5_drop02_lr5e4",
        {
            "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.tcn_encoder.SimpleTCNEncoder",
            "model.init_args.model.init_args.encoder.0.init_args": _simple_tcn_init_args(
                base_dim=64,
                dropout=0.2,
            ),
            "model.init_args.optimizer.init_args.lr": 0.0005,
            "model.init_args.optimizer.init_args.weight_decay": 0.02,
            **_FORECAST_OVERRIDES_BASE,
        },
    ),
    # ──────────────────────── ERA5 8-day forecast — TCNEncoder ────────────────────
    make_experiment(
        "era5_8df_TCN_d64_dil124_pool124_lr1e3",
        {
            "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.tcn_encoder.TCNEncoder",
            "model.init_args.model.init_args.encoder.0.init_args": _tcn_init_args_8d(),
            "model.init_args.optimizer.init_args.lr": 0.001,
            "model.init_args.optimizer.init_args.weight_decay": 0.01,
            **_FORECAST_OVERRIDES_BASE,
        },
    ),
    make_experiment(
        "era5_8df_TCN_d128_dil124_pool124_lr5e4",
        {
            "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.tcn_encoder.TCNEncoder",
            "model.init_args.model.init_args.encoder.0.init_args": _tcn_init_args_8d(
                d_model=128,
            ),
            "model.init_args.optimizer.init_args.lr": 0.0005,
            "model.init_args.optimizer.init_args.weight_decay": 0.01,
            **_FORECAST_OVERRIDES_BASE,
        },
    ),
    make_experiment(
        "era5_8df_TCN_d128_dil1248_indrop01_lr5e4",
        {
            "model.init_args.model.init_args.encoder.0.class_path": "rslearn.models.tcn_encoder.TCNEncoder",
            "model.init_args.model.init_args.encoder.0.init_args": _tcn_init_args_8d(
                d_model=128,
                dilations=[1, 2, 4, 8],
                pooling_windows=[1, 2, 8],
                dropout=0.15,
                input_dropout=0.1,
            ),
            "model.init_args.optimizer.init_args.lr": 0.0005,
            "model.init_args.optimizer.init_args.weight_decay": 0.02,
            **_FORECAST_OVERRIDES_BASE,
        },
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# LAUNCH LOGIC
# ──────────────────────────────────────────────────────────────────────────────
CONFIG_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "data",
    "helios",
    "wildfire",
    "CanadaNbac",
    "generated_configs",
    "era5d",
)


def write_config(name: str, config: dict) -> str:
    """Write a config dict to a uniquely-named YAML file and return its path."""
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.yaml"
    filepath = os.path.join(CONFIG_OUTPUT_DIR, filename)
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)

    with open(filepath, "w") as f:
        f.write(yaml_str)
    # Force flush to Weka
    os.sync()

    # Verify the write by reading back and comparing
    with open(filepath) as f:
        readback = f.read()
    if readback != yaml_str:
        raise RuntimeError(f"Config verification FAILED for {filepath}!")

    return filepath


def config_summary(config: dict) -> str:
    """Return a short summary of key hyperparameters for display."""
    enc = config["model"]["init_args"]["model"]["init_args"]["encoder"][0]["init_args"]
    opt = config["model"]["init_args"]["optimizer"]["init_args"]
    weights = config["model"]["init_args"]["model"]["init_args"]["decoders"][
        "segmentation"
    ][0]["init_args"]["weights"]
    bs = config["data"]["init_args"]["batch_size"]
    epochs = config["trainer"]["max_epochs"]
    enc_class = config["model"]["init_args"]["model"]["init_args"]["encoder"][0][
        "class_path"
    ].rsplit(".", 1)[-1]
    # Collect encoder params flexibly
    enc_params = ", ".join(
        f"{k}={v}" for k, v in enc.items() if k != "output_spatial_size"
    )
    return (
        f"  encoder={enc_class}, {enc_params}, "
        f"lr={opt['lr']}, wd={opt['weight_decay']}, weights={weights}, bs={bs}, epochs={epochs}"
    )


PROJECT_DATA_ROOT = "/weka/dfive-default/hadriens/project_data/projects"
LOCAL_DATASET_PATH = (
    "/weka/dfive-default/rslearn-eai/datasets/wildfire/canada_nbac_test"
)
RSLEARN_PYTHON = "/weka/dfive-default/hadriens/rslearn/.venv/bin/python"


def check_experiment_exists(name: str, config: dict) -> str | None:
    """Check if an experiment directory already exists.

    Returns the path if it exists, None otherwise.
    """
    project = config.get("project_name", "")
    exp_dir = os.path.join(PROJECT_DATA_ROOT, project, name)
    if os.path.exists(exp_dir):
        return exp_dir
    return None


def prepare_local_config(config: dict) -> dict:
    """Prepare a config for local rslearn model fit.

    - Swaps dataset path to the local test dataset.
    - Removes management_dir, project_name, and run_name (not needed for local rslearn fit).
    - Disables wandb logging by setting trainer.logger to false.
    """
    cfg = copy.deepcopy(config)
    cfg["data"]["init_args"]["path"] = LOCAL_DATASET_PATH
    cfg.pop("management_dir", None)
    cfg.pop("project_name", None)
    cfg.pop("run_name", None)
    cfg["trainer"]["logger"] = False
    # Remove callbacks that require a logger (e.g. LearningRateMonitor)
    if "callbacks" in cfg["trainer"]:
        cfg["trainer"]["callbacks"] = [
            cb
            for cb in cfg["trainer"]["callbacks"]
            if cb.get("class_path", "")
            != "lightning.pytorch.callbacks.LearningRateMonitor"
        ]
    return cfg


def launch_experiment(name: str, config_path: str, dry_run: bool = False) -> None:
    """Launch a single experiment via beaker_train."""
    image_name = cast(str, LAUNCH_SETTINGS["image_name"])
    cluster = cast(str, LAUNCH_SETTINGS["cluster"])
    gpus = cast(int, LAUNCH_SETTINGS["gpus"])
    priority = cast(str, LAUNCH_SETTINGS["priority"])
    wm = cast(list[dict[str, object]], LAUNCH_SETTINGS["weka_mounts"])
    import json

    cmd: list[str] = [
        "python",
        "-m",
        "rslp.main",
        "common",
        "beaker_train",
        "--image_name",
        image_name,
        f"--cluster+={cluster}",
        "--experiment_id",
        name,
        "--gpus",
        str(gpus),
        "--config_path",
        config_path,
        f"--weka_mounts+={json.dumps(wm[0])}",
        "--priority",
        priority,
    ]

    print(f"\n{'='*80}")
    print(f"  Experiment: {name}")
    print(f"  Config:     {config_path}")
    print(f"  Command:    {' '.join(cmd)}")
    print(f"{'='*80}")

    if dry_run:
        print("  [DRY RUN] Skipping launch.")
    else:
        subprocess.check_call(cmd)  # nosec B603
        print(f"  ✓ Launched: {name}")


def launch_local(name: str, config_path: str, dry_run: bool = False) -> None:
    """Run a single experiment locally via rslearn model fit."""
    cmd: list[str] = [
        RSLEARN_PYTHON,
        "-m",
        "rslearn.main",
        "model",
        "fit",
        "--config",
        config_path,
    ]
    env_info = "WANDB_MODE=disabled"

    print(f"\n{'='*80}")
    print(f"  Experiment (LOCAL): {name}")
    print(f"  Config:             {config_path}")
    print(f"  Env:                {env_info}")
    print(f"  Command:            {' '.join(cmd)}")
    print(f"{'='*80}")

    if dry_run:
        print("  [DRY RUN] Skipping local launch.")
    else:
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        subprocess.check_call(cmd, env=env)  # nosec B603
        print(f"  ✓ Finished: {name}")


def main() -> None:
    """Generate configs and launch the requested Canada NBAC ERA5 experiments."""
    parser = argparse.ArgumentParser(description="Launch CanadaFireSat experiments")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print commands without actually launching.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only launch experiments with these names.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally via 'rslearn model fit' with test dataset and no wandb.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing experiment directories if they already exist.",
    )
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [(n, c) for n, c in experiments if n in args.only]
        if not experiments:
            print(f"No experiments matched --only {args.only}")
            print(f"Available: {[n for n, _ in EXPERIMENTS]}")
            sys.exit(1)

    mode_label = "LOCAL" if args.local else "REMOTE (beaker)"

    # In local mode, prepare configs for local execution
    if args.local:
        experiments = [(n, prepare_local_config(c)) for n, c in experiments]
        print(f"  ⚙ Mode: {mode_label}")
        print(f"  ⚙ Dataset: {LOCAL_DATASET_PATH}")
        print("  ⚙ Wandb: DISABLED\n")
    else:
        # Check for duplicate experiment names (existing directories)
        conflicts = []
        for name, config in experiments:
            existing_path = check_experiment_exists(name, config)
            if existing_path:
                conflicts.append((name, existing_path))

    print(f"Will launch {len(experiments)} experiment(s) [{mode_label}]:\n")
    for name, config in experiments:
        if not args.local:
            marker = (
                " ⚠️  ALREADY EXISTS" if any(n == name for n, _ in conflicts) else ""
            )
        else:
            marker = ""
        print(f"  • {name}{marker}")
        print(config_summary(config))

    if not args.local and conflicts:
        print(f"\n{'!'*80}")
        print(
            f"  WARNING: {len(conflicts)} experiment(s) already exist and would cause checkpoint corruption:"
        )
        for cname, cpath in conflicts:
            print(f"    ✗ {cname}")
            print(f"      → {cpath}")
        print(f"{'!'*80}")

        if args.force:
            print("\n  --force specified: deleting existing experiment directories...")
            for cname, cpath in conflicts:
                shutil.rmtree(cpath)
                print(f"    ✓ Deleted: {cpath}")
        elif not args.dry_run:
            response = input(
                f"\n  Delete {len(conflicts)} existing experiment directory(ies) and continue? [y/N] "
            )
            if response.lower() == "y":
                for cname, cpath in conflicts:
                    shutil.rmtree(cpath)
                    print(f"    ✓ Deleted: {cpath}")
            else:
                print("\n  To fix this, either:")
                print("    1. Rename the experiment(s) to a unique name, or")
                print("    2. Re-run with --force to auto-delete existing directories.")
                sys.exit(1)
        else:
            print(
                "\n  [DRY RUN] Would delete these directories if not in dry-run mode."
            )
            print("  Re-run with --force (without --dry-run) to auto-delete, or")
            print("  run without --dry-run to be prompted interactively.")

    if not args.dry_run:
        response = input(
            f"\nProceed with launching {len(experiments)} experiment(s)? [y/N] "
        )
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    generated_configs: list[str] = []
    for name, config in experiments:
        config_path = write_config(name, config)
        generated_configs.append(config_path)
        if args.local:
            launch_local(name, config_path, dry_run=args.dry_run)
        else:
            launch_experiment(name, config_path, dry_run=args.dry_run)

    # Clean up generated config files for dry-run and local modes (they are
    # ephemeral and only clutter the generated_configs directory).
    if args.dry_run or args.local:
        for p in generated_configs:
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"\n{'='*80}")
    print(f"  Done! Launched {len(experiments)} experiment(s) [{mode_label}].")
    if args.dry_run or args.local:
        print(
            f"  Ephemeral configs cleaned up ({len(generated_configs)} file(s) removed)."
        )
    else:
        print(f"  Generated configs saved to: {CONFIG_OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


# cd /weka/dfive-default/hadriens/rslearn_projects
# source ./.venv/bin/activate

# Preview what will be launched:
# python -m rslp.wildfire.Canada_nbac.launch_experiments_era5 --dry-run

# Launch everything:
# python -m rslp.wildfire.Canada_nbac.launch_experiments_era5

# Launch only specific experiments:
# python -m rslp.wildfire.Canada_nbac.launch_experiments_era5 --only era5d_fullTCN_lr3e4_bs64_1pred_w005095_d5

# Run locally (test dataset, no wandb):
# python -m rslp.wildfire.Canada_nbac.launch_experiments_era5 --local --only era5d_TCN_lr1e4_d5_wd3e2_aug_mask_noise_shift
# python -m rslp.wildfire.Canada_nbac.launch_experiments_era5 --local --only era5d_fullTCN_lr5e5_bs64_w0406_d5_nocrop


# Example:
# cd /weka/dfive-default/hadriens/rslearn_projects
# source ./.venv/bin/activate
# python -m rslp.wildfire.Canada_nbac.launch_experiments_era5
