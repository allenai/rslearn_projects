"""Launch multiple Canada NBAC S2 experiments with programmatic config generation.

Each experiment is defined as a dict of overrides applied to the base config.
This avoids any issues with file-save races on Weka or stale editor buffers,
because the config is generated fresh in memory and written to a unique file
right before each launch.

Usage:
    cd /weka/dfive-default/hadriens/rslearn_projects
    source ./.venv/bin/activate
    python -m rslp.wildfire.Canada_nbac.launch_experiments_s2

    # Or run only specific experiments by name:
    python -m rslp.wildfire.Canada_nbac.launch_experiments_s2 --only exp1 exp2

    # Dry-run mode (generates configs and prints commands without launching):
    python -m rslp.wildfire.Canada_nbac.launch_experiments_s2 --dry-run

    # Local mode (runs rslearn model fit locally with test dataset, no wandb):
    python -m rslp.wildfire.Canada_nbac.launch_experiments_s2 --local
    python -m rslp.wildfire.Canada_nbac.launch_experiments_s2 --local --only exp1
    python -m rslp.wildfire.Canada_nbac.launch_experiments_s2 --local --dry-run
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
                            "class_path": "rslearn.models.olmoearth_pretrain.model.OlmoEarth",
                            "init_args": {
                                "model_id": "OLMOEARTH_V1_BASE",
                                "patch_size": 5,
                                "use_legacy_timestamps": False,
                                "token_pooling": False,
                            },
                        }
                    ],
                    "decoders": {
                        "segmentation": [
                            {
                                "class_path": "rslearn.models.attention_pooling.AttentionPool",
                                "init_args": {
                                    "in_dim": 768,
                                    "num_heads": 8,
                                    "linear_on_kv": True,
                                },
                            },
                            {
                                "class_path": "rslearn.models.layer_norm.LayerNorm",
                                "init_args": {
                                    "num_channels": 768,
                                    "normalize_dim": 1,
                                },
                            },
                            {
                                "class_path": "rslearn.models.conv.Conv",
                                "init_args": {
                                    "in_channels": 768,
                                    "out_channels": 2,
                                    "kernel_size": 2,
                                    "padding": "valid",
                                    "stride": 2,
                                    "activation": {
                                        "class_path": "torch.nn.Identity",
                                    },
                                },
                            },
                            {
                                "class_path": "rslearn.train.tasks.segmentation.SegmentationHead",
                                "init_args": {
                                    "weights": [0.05, 0.95],
                                    "dice_loss": True,
                                },
                            },
                        ]
                    },
                },
            },
            "optimizer": {
                "class_path": "rslearn.train.optimizer.AdamW",
                "init_args": {
                    "lr": 0.00004,
                    "weight_decay": 0.01,
                },
            },
            "scheduler": {
                "class_path": "rslearn.train.scheduler.CosineAnnealingScheduler",
                "init_args": {
                    "T_max": 60,
                    "eta_min_factor": 0.02,
                },
            },
        },
    },
    "data": {
        "class_path": "rslearn.train.data_module.RslearnDataModule",
        "init_args": {
            "path": "/weka/dfive-default/rslearn-eai/datasets/wildfire/canada_nbac",
            "inputs": {
                "sentinel2_l2a": {
                    "data_type": "raster",
                    "layers": ["sentinel2"],
                    "bands": [
                        "B02",
                        "B03",
                        "B04",
                        "B08",
                        "B05",
                        "B06",
                        "B07",
                        "B8A",
                        "B11",
                        "B12",
                        "B01",
                        "B09",
                    ],
                    "passthrough": True,
                    "dtype": "FLOAT32",
                    "load_all_item_groups": True,
                    "load_all_layers": True,
                },
                "label": {
                    "data_type": "raster",
                    "layers": ["label_100m"],
                    "bands": ["label_100m"],
                    "dtype": "INT32",
                    "is_target": True,
                    "resolution_factor": {
                        "class_path": "rslearn.utils.geometry.ResolutionFactor",
                        "init_args": {
                            "numerator": 1,
                            "denominator": 1,
                        },
                    },
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
            "batch_size": 16,
            "num_workers": 16,
            "default_config": {
                "crop_size": 50,
                "transforms": [
                    {
                        "class_path": "rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize",
                        "init_args": {
                            "band_names": {
                                "sentinel2_l2a": [
                                    "B02",
                                    "B03",
                                    "B04",
                                    "B08",
                                    "B05",
                                    "B06",
                                    "B07",
                                    "B8A",
                                    "B11",
                                    "B12",
                                    "B01",
                                    "B09",
                                ],
                            },
                        },
                    },
                    {
                        "class_path": "rslearn.train.transforms.resize.MaxPoolResize",
                        "init_args": {
                            "target_size": [5, 5],
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
                        "class_path": "rslearn.train.transforms.flip.Flip",
                        "init_args": {
                            "horizontal": True,
                            "vertical": True,
                            "image_selectors": [
                                "sentinel2_l2a",
                                "target/segmentation/classes",
                                "target/segmentation/valid",
                            ],
                        },
                    },
                    {
                        "class_path": "rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize",
                        "init_args": {
                            "band_names": {
                                "sentinel2_l2a": [
                                    "B02",
                                    "B03",
                                    "B04",
                                    "B08",
                                    "B05",
                                    "B06",
                                    "B07",
                                    "B8A",
                                    "B11",
                                    "B12",
                                    "B01",
                                    "B09",
                                ],
                            },
                        },
                    },
                    {
                        "class_path": "rslearn.train.transforms.resize.MaxPoolResize",
                        "init_args": {
                            "target_size": [5, 5],
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
                "center_crop": True,
            },
            "test_config": {
                "groups": ["test"],
                "load_all_crops": True,
            },
            "predict_config": {
                "load_all_crops": True,
                "overlap_pixels": 5,
                "skip_targets": True,
            },
        },
    },
    "trainer": {
        "max_epochs": 60,
        "logger": {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": {
                "group": "S2_only",
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
            {
                "class_path": "rslearn.train.callbacks.freeze_unfreeze.MultiStageFineTuning",
                "init_args": {
                    "stages": [
                        {
                            "at_epoch": 0,
                            "freeze_selectors": ["model.encoder.0"],
                            "unfreeze_selectors": [],
                        },
                        {
                            "at_epoch": 15,
                            "freeze_selectors": [],
                            "unfreeze_selectors": ["model.encoder.0"],
                            "scale_existing_groups": 0.1,
                            "unfreeze_lr_factor": 1.0,
                        },
                    ],
                },
            },
        ],
    },
    "management_dir": "${RSLP_PREFIX}/projects",
    "project_name": "20260329_wf_nbac_newsample_LF",
    "run_name": "placeholder",  # overridden per experiment
}


# ──────────────────────────────────────────────────────────────────────────────
# AUGMENTATION TRANSFORM FACTORIES
#
# Each function returns a transform dict.  Fixed pipeline steps (flip,
# normalize, resize) have no parameters; augmentation steps expose the knobs
# you're most likely to sweep over.
# ──────────────────────────────────────────────────────────────────────────────
S2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]
TARGET_SELECTORS = ["target/segmentation/classes", "target/segmentation/valid"]


def _flip() -> dict:
    """Spatial flip (horizontal + vertical)."""
    return {
        "class_path": "rslearn.train.transforms.flip.Flip",
        "init_args": {
            "horizontal": True,
            "vertical": True,
            "image_selectors": ["sentinel2_l2a"] + TARGET_SELECTORS,
        },
    }


def _normalize() -> dict:
    """OlmoEarth band normalization."""
    return {
        "class_path": "rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize",
        "init_args": {
            "band_names": {"sentinel2_l2a": S2_BANDS},
        },
    }


def _resize(target_size: list[int] = [5, 5]) -> dict:
    """Downsample target masks to match encoder output using max-pooling.

    MaxPoolResize preserves positive pixels: if *any* source pixel in a
    pooling window is 1, the output pixel is 1.  The previous nearest-
    neighbour Resize could silently drop rare fire pixels.
    """
    return {
        "class_path": "rslearn.train.transforms.resize.MaxPoolResize",
        "init_args": {
            "target_size": target_size,
            "selectors": TARGET_SELECTORS,
        },
    }


def _time_drop(drop_ratio: float = 0.2, min_keep: int = 2) -> dict:
    """RandomTimeDropping — drop a fraction of S2 timesteps."""
    return {
        "class_path": "rslearn.train.transforms.random_time_dropping.RandomTimeDropping",
        "init_args": {
            "drop_ratio": drop_ratio,
            "min_keep": min_keep,
            "selectors": ["sentinel2_l2a"],
        },
    }


def _noise(std: float = 0.02) -> dict:
    """GaussianNoise — additive noise on normalized S2 bands."""
    return {
        "class_path": "rslearn.train.transforms.gaussian_noise.GaussianNoise",
        "init_args": {
            "std": std,
            "selectors": ["sentinel2_l2a"],
        },
    }


def _train_transforms(*augments: dict) -> list[dict]:
    """Build train_config transforms: flip → normalize → [augments] → resize."""
    return [_flip(), _normalize()] + list(augments) + [_resize()]


# ──────────────────────────────────────────────────────────────────────────────
# LAUNCH SETTINGS — shared across all experiments
# ──────────────────────────────────────────────────────────────────────────────
LAUNCH_SETTINGS = {
    "image_name": "hadriens/rslpomp_hspec_260408_wbfix",
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


# ──────────────────────────────── YOUR EXPERIMENTS ─────────────────────────────
# Edit this list to define your experiment variants.
# Each entry is: make_experiment("experiment_name", {dotted.key.path: value, ...})
#
# Example overrides you can use:
#   "model.init_args.optimizer.init_args.lr": 0.00002
#   "model.init_args.model.init_args.encoder.0.init_args.use_legacy_timestamps": True
#   "model.init_args.model.init_args.encoder.0.init_args.token_pooling": True
#   "model.init_args.model.init_args.decoders.segmentation.3.init_args.weights": [0.1, 0.9]
#   "model.init_args.model.init_args.decoders.segmentation.3.init_args.loss_weights": [1.0, 1.0]
#   "model.init_args.model.init_args.decoders.segmentation.3.init_args.dice_loss": False
#   "model.init_args.model.init_args.decoders.segmentation.3.init_args.focal_loss": True
#   "model.init_args.model.init_args.decoders.segmentation.3.init_args.focal_loss_alpha": 0.8
#   "data.init_args.batch_size": 32
#   "trainer.max_epochs": 60

EXPERIMENTS = [
    # ── Augmentation ablation (same base recipe, vary augmentation) ──────────
    # make_experiment("s2_mts_atp_lr5e4_cs5_bs16_aug_none_randinit", {
    #     "data.init_args.train_config.transforms": _train_transforms(),
    #     "model.init_args.model.init_args.encoder.0.init_args.random_initialization": True,
    # }),
    # make_experiment("s2_mts_atp_lr1e5_cs5_bs16_aug_tdrop05_lrfixed", {
    #     "data.init_args.train_config.transforms": _train_transforms(_time_drop(0.5)),
    #     "model.init_args.optimizer.init_args.lr": 1e-5,
    # }),
    # make_experiment("s2_mts_atp_ln_lr1e5_ps10_cs100_bs16_tdrop05_lnorm", {
    #     # --- Encoder: patch_size 10 instead of 5 ---
    #     "model.init_args.model.init_args.encoder.0.init_args.patch_size": 10,
    #     # --- Optimizer ---
    #     "model.init_args.optimizer.init_args.lr": 1e-5,
    #     # --- Crop / resize: 100 px crop → 10×10 tokens (100/10) ---
    #     "data.init_args.default_config": {
    #         "crop_size": 100,
    #         "transforms": [
    #             _normalize(),
    #             _resize([10, 10]),
    #         ],
    #     },
    #     "data.init_args.train_config.transforms": [
    #         _flip(), _normalize(), _time_drop(0.5), _resize([10, 10]),
    #     ],
    #     # --- Decoder: AttentionPool → LayerNorm → Conv(1×1, stride 1) → SegHead ---
    #     "model.init_args.model.init_args.decoders.segmentation": [
    #         {
    #             "class_path": "rslearn.models.attention_pooling.AttentionPool",
    #             "init_args": {
    #                 "in_dim": 768,
    #                 "num_heads": 8,
    #                 "linear_on_kv": True,
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.models.layer_norm.LayerNorm",
    #             "init_args": {
    #                 "num_channels": 768,
    #                 "normalize_dim": 1,
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.models.conv.Conv",
    #             "init_args": {
    #                 "in_channels": 768,
    #                 "out_channels": 2,
    #                 "kernel_size": 1,
    #                 "padding": "same",
    #                 "stride": 1,
    #                 "activation": {
    #                     "class_path": "torch.nn.Identity",
    #                 },
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.train.tasks.segmentation.SegmentationHead",
    #             "init_args": {
    #                 "weights": [0.05, 0.95],
    #                 "dice_loss": True,
    #             },
    #         },
    #     ],
    # }),
    # make_experiment("s2_mts_atp_lr5e6_cs5_bs16_aug_tdrop05", {
    #     "data.init_args.train_config.transforms": _train_transforms(_time_drop(0.5)),
    #     "model.init_args.optimizer.init_args.lr": 5e-6,
    # }),
    # make_experiment("s2_mts_atp_ln_lr1e5_cs5_bs16_tdrop05_lnorm", {
    #     "data.init_args.train_config.transforms": _train_transforms(_time_drop(0.5)),
    #     "model.init_args.optimizer.init_args.lr": 1e-5,
    #     "model.init_args.model.init_args.decoders.segmentation": [
    #         {
    #             "class_path": "rslearn.models.attention_pooling.AttentionPool",
    #             "init_args": {
    #                 "in_dim": 768,
    #                 "num_heads": 8,
    #                 "linear_on_kv": True,
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.models.layer_norm.LayerNorm",
    #             "init_args": {
    #                 "num_channels": 768,
    #                 "normalize_dim": 1,
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.models.conv.Conv",
    #             "init_args": {
    #                 "in_channels": 768,
    #                 "out_channels": 2,
    #                 "kernel_size": 2,
    #                 "padding": "valid",
    #                 "stride": 2,
    #                 "activation": {
    #                     "class_path": "torch.nn.Identity",
    #                 },
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.train.tasks.segmentation.SegmentationHead",
    #             "init_args": {
    #                 "weights": [0.05, 0.95],
    #                 "dice_loss": True,
    #             },
    #         }
    #     ],
    # }),
    # make_experiment("s2_mts_atp_ln_lr1e6_cs5_bs16_tdrop05_lnorm", {
    #     "data.init_args.train_config.transforms": _train_transforms(_time_drop(0.5)),
    #     "model.init_args.optimizer.init_args.lr": 1e-6,
    #     "model.init_args.model.init_args.decoders.segmentation": [
    #         {
    #             "class_path": "rslearn.models.attention_pooling.AttentionPool",
    #             "init_args": {
    #                 "in_dim": 768,
    #                 "num_heads": 8,
    #                 "linear_on_kv": True,
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.models.layer_norm.LayerNorm",
    #             "init_args": {
    #                 "num_channels": 768,
    #                 "normalize_dim": 1,
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.models.conv.Conv",
    #             "init_args": {
    #                 "in_channels": 768,
    #                 "out_channels": 2,
    #                 "kernel_size": 2,
    #                 "padding": "valid",
    #                 "stride": 2,
    #                 "activation": {
    #                     "class_path": "torch.nn.Identity",
    #                 },
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.train.tasks.segmentation.SegmentationHead",
    #             "init_args": {
    #                 "weights": [0.05, 0.95],
    #                 "dice_loss": True,
    #             },
    #         }
    #     ],
    # }),
    # make_experiment("s2_mts_atp_ln_lr5e6_cs8_bs16_tdrop05_lnorm_60ep", {
    #     "model.init_args.optimizer.init_args.lr": 5e-6,
    #     "data.init_args.default_config": {
    #         "crop_size": 80,
    #         "transforms": [_normalize(), _resize([8, 8])],
    #     },
    #     "data.init_args.train_config.transforms": [
    #         _flip(), _normalize(), _resize([8, 8]),
    #     ],
    # }),
    # make_experiment(
    #     "s2_mts_atp_ln_lr5e6_cs10_bs16_tdrop05_lnorm_60ep",
    #     {
    #         "model.init_args.optimizer.init_args.lr": 5e-6,
    #         "data.init_args.default_config": {
    #             "crop_size": 100,
    #             "transforms": [_normalize(), _resize([10, 10])],
    #         },
    #         "data.init_args.train_config.transforms": [
    #             _flip(),
    #             _normalize(),
    #             _resize([10, 10]),
    #         ],
    #     },
    # ),
    # make_experiment("s2_mts_atp_ln_lr5e6_cs5_bs16_tdrop05_lnorm_ep60_fbs16", {
    #     "data.init_args.train_config.transforms": _train_transforms(_time_drop(0.5)),
    #     "model.init_args.optimizer.init_args.lr": 5e-6,
    #     "model.init_args.scheduler.init_args.eta_min_factor": 0.025,
    #     "trainer.max_epochs": 60,
    #     "model.init_args.scheduler.init_args.T_max": 60,
    # }),
    make_experiment(
        "subsample_oep_eval_base_lr1e5_v2",
        {
            "data.init_args.train_config.tags": {"oep_eval": ""},
            "data.init_args.train_config.transforms": _train_transforms(
                _time_drop(0.5)
            ),
            "data.init_args.val_config.tags": {"oep_eval": ""},
            "model.init_args.optimizer.init_args.lr": 1e-5,
            "model.init_args.scheduler.init_args.T_max": 40,
            "model.init_args.scheduler.init_args.eta_min_factor": 0.05,
            "trainer.max_epochs": 40,
        },
    ),
    # make_experiment("s2_mts_atp_ln_lr5e6_cs5_bs16_tdrop05_lnorm_wd01", {
    #     "data.init_args.train_config.transforms": _train_transforms(_time_drop(0.5)),
    #     "model.init_args.optimizer.init_args.lr": 5e-6,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.1,
    #     "model.init_args.model.init_args.decoders.segmentation": [
    #         {
    #             "class_path": "rslearn.models.attention_pooling.AttentionPool",
    #             "init_args": {
    #                 "in_dim": 768,
    #                 "num_heads": 8,
    #                 "linear_on_kv": True,
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.models.layer_norm.LayerNorm",
    #             "init_args": {
    #                 "num_channels": 768,
    #                 "normalize_dim": 1,
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.models.conv.Conv",
    #             "init_args": {
    #                 "in_channels": 768,
    #                 "out_channels": 2,
    #                 "kernel_size": 2,
    #                 "padding": "valid",
    #                 "stride": 2,
    #                 "activation": {
    #                     "class_path": "torch.nn.Identity",
    #                 },
    #             },
    #         },
    #         {
    #             "class_path": "rslearn.train.tasks.segmentation.SegmentationHead",
    #             "init_args": {
    #                 "weights": [0.05, 0.95],
    #                 "dice_loss": True,
    #             },
    #         }
    #     ],
    # }),
    # make_experiment("s2_mts_atp_lr1e5_cs5_bs16_aug_tdrop05_wd007", {
    #     "data.init_args.train_config.transforms": _train_transforms(_time_drop(0.5)),
    #     "model.init_args.optimizer.init_args.lr": 1e-5,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.07,
    # }),
    # make_experiment("s2_mts_atp_lr1e5_cs5_bs16_aug_tdrop05_wd01", {
    #     "data.init_args.train_config.transforms": _train_transforms(_time_drop(0.5)),
    #     "model.init_args.optimizer.init_args.lr": 1e-5,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.1,
    # }),
    #     make_experiment("s2_mts_atp_lr1e5_cs5_bs16_aug_tdrop05_wd03", {
    #     "data.init_args.train_config.transforms": _train_transforms(_time_drop(0.5)),
    #     "model.init_args.optimizer.init_args.lr": 1e-5,
    #     "model.init_args.optimizer.init_args.weight_decay": 0.3,
    # }),
    # make_experiment("s2_mts_atp_lr5e4_cs5_bs16_aug_gnoise002", {
    #     "data.init_args.train_config.transforms": _train_transforms(_noise()),
    # }),
    # make_experiment("s2_mts_atp_lr5e4_cs5_bs16_aug_tdrop02_gnoise002", {
    #     "data.init_args.train_config.transforms": _train_transforms(
    #         _time_drop(), _noise()
    #     ),
    # }),
    # ── Previous experiments (loss / focal ablation) ─────────────────────────
    # make_experiment("s2_mts_atp_cs5_bs16_lr5e4_wd1e2_lw0515_focal", {
    #     "model.init_args.model.init_args.decoders.segmentation.3.init_args.loss_weights": [0.5, 1.5],
    #     "model.init_args.model.init_args.decoders.segmentation.3.init_args.focal_loss": True,
    # }),
    # make_experiment("s2_mts_atp_cs5_bs16_lr5e4_wd1e2_lw0515_focal_fa08", {
    #     "model.init_args.model.init_args.decoders.segmentation.3.init_args.loss_weights": [0.5, 1.5],
    #     "model.init_args.model.init_args.decoders.segmentation.3.init_args.focal_loss": True,
    #     "model.init_args.model.init_args.decoders.segmentation.3.init_args.focal_loss_alpha": 0.8,
    # }),
    # make_experiment("s2_mts_atp_cs5_bs16_lr5e4_wd1e2", {
    # }),
    # make_experiment("s2_mts_atp_cs5_bs16_lr5e4_wd1e2_lw0515", {
    #     "model.init_args.model.init_args.decoders.segmentation.3.init_args.loss_weights": [0.5, 1.5]
    # }),
    # make_experiment("s2_mts_atpool_nodice_focal", {
    #     "model.init_args.model.init_args.decoders.segmentation.3.init_args.dice_loss": False,
    #     "model.init_args.model.init_args.decoders.segmentation.3.init_args.focal_loss": True,
    #     "model.init_args.model.init_args.decoders.segmentation.3.init_args.focal_loss_alpha": 0.75,
    # }),
    # make_experiment("s2_mts_atpool_tokenpool_bs8", {
    #     "model.init_args.model.init_args.encoder.0.init_args.token_pooling": True,
    #     "data.init_args.batch_size": 8,
    # }),
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
    "S2",
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
    seg_head = config["model"]["init_args"]["model"]["init_args"]["decoders"][
        "segmentation"
    ][-1]["init_args"]
    bs = config["data"]["init_args"]["batch_size"]
    epochs = config["trainer"]["max_epochs"]
    weights = seg_head.get("weights")
    loss_weights = seg_head.get("loss_weights", seg_head.get("loss_weigths"))
    dice_loss = seg_head.get("dice_loss")
    focal_loss = seg_head.get("focal_loss", False)
    return (
        f"  lr={opt['lr']}, wd={opt['weight_decay']}, legacy_ts={enc.get('use_legacy_timestamps')}, "
        f"token_pooling={enc.get('token_pooling')}, weights={weights}, loss_weights={loss_weights}, "
        f"dice_loss={dice_loss}, focal_loss={focal_loss}, bs={bs}, epochs={epochs}"
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
    """Generate configs and launch the requested Canada NBAC S2 experiments."""
    parser = argparse.ArgumentParser(description="Launch Canada NBAC S2 experiments")
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
# python -m rslp.wildfire.Canada_nbac.launch_experiments_s2 --dry-run

# Launch everything:
# python -m rslp.wildfire.Canada_nbac.launch_experiments_s2

# Launch only specific experiments:
# python -m rslp.wildfire.Canada_nbac.launch_experiments_s2 --only s2_mts_atp_cs5_bs16_lr5e4_wd1e2_lw0515

# Run locally (test dataset, no wandb):
# python -m rslp.wildfire.Canada_nbac.launch_experiments_s2 --local
# python -m rslp.wildfire.Canada_nbac.launch_experiments_s2 --local --only exp1 --dry-run

# Example:
# cd /weka/dfive-default/hadriens/rslearn_projects
# source ./.venv/bin/activate
# python -m rslp.wildfire.Canada_nbac.launch_experiments_s2
