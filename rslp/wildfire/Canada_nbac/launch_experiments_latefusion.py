"""Launch multiple CanadaFireSat late-fusion (S2 + ERA5) experiments.

Each experiment is defined as a dict of overrides applied to the base config.
This avoids any issues with file-save races on Weka or stale editor buffers,
because the config is generated fresh in memory and written to a unique file
right before each launch.
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
# ERA5 per-channel normalization stats (from era5d_norm_stats.json)
# Band order: d2m, e, pev, ro, sp, ssr, ssrd, str, swvl1, swvl2, t2m, tp, u10, v10
# ──────────────────────────────────────────────────────────────────────────────
ERA5_MEAN = [
    268.209,
    -0.00113,
    -0.00833,
    0.000897,
    94745.2,
    9079014.2,
    11716921.0,
    -4499458.3,
    0.343,
    0.339,
    -0.0584,
    0.00188,
    0.463,
    0.0691,
]
ERA5_STD = [
    14.149,
    0.00125,
    0.00966,
    0.00221,
    5192.3,
    7486798.9,
    8479277.6,
    2228211.3,
    0.1436,
    0.1426,
    13.825,
    0.00425,
    1.787,
    1.697,
]


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
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


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM FACTORIES
# ──────────────────────────────────────────────────────────────────────────────
def _flip() -> dict:
    """Spatial flip (horizontal + vertical) — S2 and targets only."""
    return {
        "class_path": "rslearn.train.transforms.flip.Flip",
        "init_args": {
            "horizontal": True,
            "vertical": True,
            "image_selectors": ["sentinel2_l2a"] + TARGET_SELECTORS,
        },
    }


def _olmo_normalize() -> dict:
    """OlmoEarth band normalization for Sentinel-2."""
    return {
        "class_path": "rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize",
        "init_args": {
            "band_names": {"sentinel2_l2a": S2_BANDS},
        },
    }


def _era5_normalize() -> dict:
    """Channel-wise normalization for ERA5 daily variables."""
    return {
        "class_path": "rslearn.train.transforms.normalize.Normalize",
        "init_args": {
            "mean": ERA5_MEAN,
            "std": ERA5_STD,
            "selectors": ["era5_daily"],
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


def _time_drop_s2(drop_ratio: float = 0.2, min_keep: int = 2) -> dict:
    """RandomTimeDropping — drop a fraction of S2 timesteps."""
    return {
        "class_path": "rslearn.train.transforms.random_time_dropping.RandomTimeDropping",
        "init_args": {
            "drop_ratio": drop_ratio,
            "min_keep": min_keep,
            "selectors": ["sentinel2_l2a"],
        },
    }


def _noise_s2(std: float = 0.02) -> dict:
    """GaussianNoise — additive noise on normalized S2 bands."""
    return {
        "class_path": "rslearn.train.transforms.gaussian_noise.GaussianNoise",
        "init_args": {
            "std": std,
            "selectors": ["sentinel2_l2a"],
        },
    }


def _era5_mask(mask_ratio: float = 0.10) -> dict:
    """RandomTimeMasking — zero-out a fraction of ERA5 timesteps."""
    return {
        "class_path": "rslearn.train.transforms.ts_augment.RandomTimeMasking",
        "init_args": {
            "mask_ratio": mask_ratio,
            "selectors": ["era5_daily"],
        },
    }


def _era5_shift(max_shift: int = 2) -> dict:
    """TemporalShift — shift ERA5 timesteps by ±max_shift."""
    return {
        "class_path": "rslearn.train.transforms.ts_augment.TemporalShift",
        "init_args": {
            "max_shift": max_shift,
            "selectors": ["era5_daily"],
        },
    }


def _era5_noise(std: float = 0.02) -> dict:
    """GaussianNoise — additive noise on normalized ERA5 bands."""
    return {
        "class_path": "rslearn.train.transforms.gaussian_noise.GaussianNoise",
        "init_args": {
            "std": std,
            "selectors": ["era5_daily"],
        },
    }


def _seg_head() -> dict:
    """Standard segmentation head configuration (shared by all decoder configs)."""
    return {
        "class_path": "rslearn.train.tasks.segmentation.SegmentationHead",
        "init_args": {
            "weights": [0.05, 0.95],
            "dice_loss": True,
        },
    }


def _shallow_decoder(
    in_channels: int = 896,
    context_key: str | None = None,
) -> list[dict]:
    """Single-Conv decoder: in_ch → 2, k=2, s=2, no activation (linear).

    This is the original decoder used in early experiments.
    """
    conv_init_args = {
        "in_channels": in_channels,
        "out_channels": 2,
        "kernel_size": 2,
        "padding": "valid",
        "stride": 2,
        "activation": {
            "class_path": "torch.nn.Identity",
        },
    }
    if context_key is not None:
        conv_init_args["context_key"] = context_key

    return [
        {
            "class_path": "rslearn.models.conv.Conv",
            "init_args": conv_init_args,
        },
        _seg_head(),
    ]


def _deep_decoder(in_channels: int = 896, hidden_channels: int = 256) -> list[dict]:
    """Two-layer decoder: in_ch → hidden (ReLU) → 2, k=2, s=2.

    The intermediate ReLU allows the model to learn nonlinear interactions
    between S2 and ERA5 features (e.g. "fuel AND dry → fire"), which a
    single linear Conv cannot express.
    """
    return [
        {
            "class_path": "rslearn.models.conv.Conv",
            "init_args": {
                "in_channels": in_channels,
                "out_channels": hidden_channels,
                "kernel_size": 1,
                "padding": "same",
                # default activation is ReLU(inplace=True)
            },
        },
        {
            "class_path": "rslearn.models.conv.Conv",
            "init_args": {
                "in_channels": hidden_channels,
                "out_channels": 2,
                "kernel_size": 2,
                "padding": "valid",
                "stride": 2,
                "activation": {
                    "class_path": "torch.nn.Identity",
                },
            },
        },
        _seg_head(),
    ]


def _default_transforms() -> list[dict]:
    """Default (val/test) transforms: ERA5 normalize → S2 normalize → resize."""
    return [
        _era5_normalize(),
        _olmo_normalize(),
        _resize(),
    ]


def _train_transforms(
    era5_mask_ratio: float = 0.10,
    *extra_augments: dict,
) -> list[dict]:
    """Build train_config transforms.

    Default augmentations (proven on independent encoders):
      - S2:   RandomTimeDropping (drop_ratio=0.5, min_keep=2)
      - ERA5: RandomTimeMasking  (mask_ratio) + TemporalShift (max_shift=2)
    """
    return (
        [
            _flip(),
            _era5_normalize(),
            _olmo_normalize(),
            _time_drop_s2(drop_ratio=0.5, min_keep=2),  # S2
            _era5_mask(mask_ratio=era5_mask_ratio),  # ERA5
            _era5_shift(max_shift=2),  # ERA5
        ]
        + list(extra_augments)
        + [
            _resize(),
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# BASE CONFIG — this is the "template" matching finetune_s2_era5_latefusion.yaml
# plus the ERA5 transforms (unflatten + normalize) which the YAML was missing.
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
                            "class_path": "rslearn.models.xatt_fusion.CrossAttentionFusionExtractor",
                            "init_args": {
                                "primary_path": [
                                    {
                                        "class_path": "rslearn.models.olmoearth_pretrain.model.OlmoEarth",
                                        "init_args": {
                                            "model_id": "OLMOEARTH_V1_BASE",
                                            "patch_size": 5,
                                            "use_legacy_timestamps": False,
                                            "token_pooling": False,
                                        },
                                    },
                                    {
                                        "class_path": "rslearn.models.attention_pooling.AttentionPool",
                                        "init_args": {
                                            "in_dim": 768,
                                            "num_heads": 8,
                                            "linear_on_kv": True,
                                        },
                                    },
                                ],
                                "context_paths": [
                                    # ── Context path 0: ERA5 daily via TCNEncoder ──
                                    [
                                        {
                                            "class_path": "rslearn.models.tcn_encoder.TCNEncoder",
                                            "init_args": {
                                                "in_channels": 14,
                                                "d_model": 128,
                                                "d_output": 128,
                                                "dilations": [
                                                    1,
                                                    2,
                                                    4,
                                                    8,
                                                    16,
                                                    32,
                                                    64,
                                                    128,
                                                ],
                                                "pooling_windows": [1, 2, 4, 12],
                                                "dropout": 0.1,
                                                "output_spatial_size": 10,
                                            },
                                        },
                                    ],
                                ],
                                "primary_output_channels": 768,
                                "context_output_channels": [128],
                            },
                        }
                    ],
                    "decoders": {
                        "segmentation": [
                            {
                                "class_path": "rslearn.models.conv.Conv",
                                "init_args": {
                                    "in_channels": 896,  # 768 (S2) + 128 (ERA5)
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
                        ],
                    },
                },
            },
            "optimizer": {
                "class_path": "rslearn.train.optimizer.AdamW",
                "init_args": {
                    "lr": 1e-6,
                    "weight_decay": 0.01,
                },
            },
            "scheduler": {
                "class_path": "rslearn.train.scheduler.CosineAnnealingScheduler",
                "init_args": {
                    "T_max": 40,
                    "eta_min_factor": 0.05,
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
                    "bands": S2_BANDS,
                    "passthrough": True,
                    "dtype": "FLOAT32",
                    "load_all_item_groups": True,
                    "load_all_layers": True,
                },
                "era5_daily": {
                    "data_type": "raster",
                    "layers": ["era5_365dhistory"],
                    "use_all_bands_in_order_of_band_set_idx": 0,
                    "passthrough": True,
                    "dtype": "FLOAT32",
                    "resolution_factor": {
                        "class_path": "rslearn.utils.geometry.ResolutionFactor",
                        "init_args": {
                            "numerator": 1,
                            "denominator": 260,
                        },
                    },
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
                "transforms": _default_transforms(),
            },
            "train_config": {
                "groups": ["train"],
                "transforms": _train_transforms(),
            },
            "val_config": {
                "groups": ["val"],
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
        "max_epochs": 40,
        "logger": {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": {
                "group": "late_fusion",
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
            # Epochs 0–9:  OlmoEarth frozen, ERA5 TCN + fusion + decoder trainable.
            # Epoch 10+:   unfreeze entire OlmoEarth, scale existing LR by 0.333.
            # model.encoder.0           = CrossAttentionFusionExtractor
            # model.encoder.0.paths.0   = primary path / S2 (ModuleList)
            # model.encoder.0.paths.0.0 = OlmoEarth  ← freeze target
            {
                "class_path": "rslearn.train.callbacks.freeze_unfreeze.MultiStageFineTuning",
                "init_args": {
                    "stages": [
                        {
                            "at_epoch": 0,
                            "freeze_selectors": ["model.encoder.0.paths.0.0"],
                            "unfreeze_selectors": [],
                        },
                        {
                            "at_epoch": 10,
                            "freeze_selectors": [],
                            "unfreeze_selectors": ["model.encoder.0.paths.0.0"],
                            "scale_existing_groups": 0.333,
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
# Useful override paths:
#   "model.init_args.optimizer.init_args.lr": 0.00002
#   "model.init_args.optimizer.init_args.weight_decay": 0.03
#   "model.init_args.model.init_args.encoder.0.init_args.post_fusion_mode": "self_attn_ffn"
#   "model.init_args.model.init_args.encoder.0.init_args.context_dropout_prob": 0.15
#   "model.init_args.model.init_args.encoder.0.init_args.context_paths.0.0.init_args.dropout": 0.3
#   "model.init_args.model.init_args.encoder.0.init_args.context_paths.0.0.init_args.d_model": 256
#   "model.init_args.model.init_args.decoders.segmentation.1.init_args.weights": [0.1, 0.9]
#   "model.init_args.model.init_args.decoders.segmentation.1.init_args.dice_loss": False
#   "data.init_args.batch_size": 8
#   "data.init_args.train_config.transforms": _train_transforms(_time_drop_s2(), _era5_mask())
#   "trainer.max_epochs": 60

# ──────────────────────────────────────────────────────────────────────────────
# ERA5 warm-start checkpoints (best standalone ERA5-only runs)
# ──────────────────────────────────────────────────────────────────────────────

# TCN: TCNEncoder(d_model=128, d_output=2) — d_output differs from
# late-fusion (d_output=128), so the MLP layers are shape-incompatible and
# must be ignored.  The backbone (input_norm, input_proj, tcn_blocks,
# pooling_modules) is fully compatible.
ERA5_STANDALONE_CKPT = (
    "/weka/dfive-default/hadriens/project_data/projects/"
    "20260321_wf_nbac_newsample/"
    "era5d_TCN_lr5e5_d5_wd3e2_aug_mask_shift_long2/"
    "checkpoints/epoch=20-step=46998.ckpt"
)

# XFMR: PatchTransformerEncoder(d_model=192, d_output=2) — d_output differs
# from late-fusion (d_output=128), so the head layers are shape-incompatible
# and must be ignored.  The backbone (patch_embed, time_embed, blocks,
# final_norm, attn_query) is fully compatible.
ERA5_XFMR_STANDALONE_CKPT = (
    "/weka/dfive-default/hadriens/project_data/projects/"
    "20260321_wf_nbac_newsample/"
    "era5d_xfmr_base_14x7_attn_sin_doy_rel_lr1e5/"
    "checkpoints/epoch=7-step=17904.ckpt"
)


def _era5_warm_restore_config() -> dict:
    """RestoreConfig that warm-starts the ERA5 TCN backbone from a standalone checkpoint.

    Key mapping (after selector drills into the Lightning state_dict):
        source:  model.encoder.0.<param>          (ERA5-only model)
        target:  encoder.0.paths.1.0.<param>      (late-fusion ERA5 path)

    Ignored layers (shape-incompatible, d_output=2 vs 128):
        model.encoder.0.mlp.*

    Also ignored:
        model.decoders.*   (standalone segmentation head, not used in late-fusion)
    """
    return {
        "class_path": "rslearn.train.lightning_module.RestoreConfig",
        "init_args": {
            "restore_path": ERA5_STANDALONE_CKPT,
            "selector": ["state_dict"],
            "ignore_prefixes": [
                "model.decoders.",  # standalone decoder (different structure)
                "model.encoder.0.mlp.",  # MLP head (d_output=2 vs 128)
            ],
            "remap_prefixes": [
                ["model.encoder.0.", "encoder.0.paths.1.0."],
            ],
        },
    }


def _era5_xfmr_warm_restore_config() -> dict:
    """RestoreConfig that warm-starts the ERA5 PatchTransformerEncoder from a standalone checkpoint.

    Key mapping (after selector drills into the Lightning state_dict):
        source:  model.encoder.0.<param>          (ERA5-only model)
        target:  encoder.0.paths.1.0.<param>      (late-fusion ERA5 path)

    Ignored layers (shape-incompatible, d_output=2 vs 128):
        model.encoder.0.head.*

    Also ignored:
        model.decoders.*   (standalone segmentation head, not used in late-fusion)
    """
    return {
        "class_path": "rslearn.train.lightning_module.RestoreConfig",
        "init_args": {
            "restore_path": ERA5_XFMR_STANDALONE_CKPT,
            "selector": ["state_dict"],
            "ignore_prefixes": [
                "model.decoders.",
                "model.encoder.0.head.",
            ],
            "remap_prefixes": [
                ["model.encoder.0.", "encoder.0.paths.1.0."],
            ],
        },
    }


def _xfmr_era5_path(d_output: int = 256, output_spatial_size: int = 10) -> list[dict]:
    """Build the ERA5 branch using PatchTransformerEncoder instead of TCNEncoder."""
    return [
        {
            "class_path": "rslearn.models.transformer_encoder.PatchTransformerEncoder",
            "init_args": {
                "in_channels": 14,
                "d_model": 192,
                "d_output": d_output,
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
                "output_spatial_size": output_spatial_size,
                "has_mask_channel": False,
                "pad_value": 0.0,
            },
        },
    ]


def _xfmr_logitres_era5_path(output_spatial_size: int = 10) -> list[dict]:
    """Build the ERA5 branch for logit-residual fusion using PatchTransformerEncoder.

    Like _logitres_era5_path but with a XFMR backbone instead of TCN. Output is
    projected from d_output=256 to 768 via a 1x1 Conv to match the S2 path.
    """
    return _xfmr_era5_path(d_output=256, output_spatial_size=output_spatial_size) + [
        {
            "class_path": "rslearn.models.conv.Conv",
            "init_args": {
                "in_channels": 256,
                "out_channels": 768,
                "kernel_size": 1,
                "padding": "same",
                "activation": {
                    "class_path": "torch.nn.Identity",
                },
            },
        },
    ]


def _fusion_path_bottleneck_conv(in_channels: int, out_channels: int) -> list[dict]:
    """Single 1×1 same-padding Conv: backbone channels → fusion width, then ReLU.

    One linear projection per path (not a two-layer MLP). Appended after each backbone
    so cross-attention sees ``out_channels`` per path.
    """
    return [
        {
            "class_path": "rslearn.models.conv.Conv",
            "init_args": {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": 1,
                "padding": "same",
                "activation": {
                    "class_path": "torch.nn.ReLU",
                    "init_args": {"inplace": True},
                },
            },
        },
    ]


def _xattn_encoder_block_xfmr(
    post_fusion_mode: str,
    fusion_bottleneck_dim: int | None = None,
    *,
    s2_backbone_channels: int = 768,
    era5_backbone_channels: int = 256,
) -> dict:
    """Like _xattn_encoder_block but with XFMR ERA5 path instead of TCN.

    If ``fusion_bottleneck_dim`` is set, appends one 1×1 Conv + ReLU after each path
    (OlmoEarth stack and ERA5 XFMR) so fusion runs at that width; decoder should use
    the same channel count. Freeze selectors target only backbone modules
    (``paths.0.0``, ``paths.1.0``), so these projections train whenever the optimizer runs.
    """
    base_init = cast(
        dict[str, Any],
        cast(
            list[dict[str, Any]],
            cast(dict[str, Any], BASE_CONFIG["model"])["init_args"]["model"][
                "init_args"
            ]["encoder"],
        )[0]["init_args"],
    )
    primary_path = copy.deepcopy(base_init["primary_path"])
    context_path = _xfmr_era5_path(d_output=era5_backbone_channels)

    if fusion_bottleneck_dim is None:
        primary_out = s2_backbone_channels
        context_out = era5_backbone_channels
    else:
        d = fusion_bottleneck_dim
        primary_path = primary_path + _fusion_path_bottleneck_conv(
            s2_backbone_channels, d
        )
        context_path = context_path + _fusion_path_bottleneck_conv(
            era5_backbone_channels, d
        )
        primary_out = d
        context_out = d

    return {
        "class_path": "rslearn.models.xatt_fusion.CrossAttentionFusionExtractor",
        "init_args": {
            "primary_path": primary_path,
            "context_paths": [context_path],
            "primary_output_channels": primary_out,
            "context_output_channels": [context_out],
            "attention_dim": 256,
            "num_memory_tokens": 4,
            "num_heads": 4,
            "attention_dropout": 0.1,
            "residual_dropout": 0.0,
            "post_fusion_mode": post_fusion_mode,
            "ffn_expansion": 2.0,
            "ffn_activation": "swiglu",
            "ffn_dropout": 0.1,
            "pre_fusion_dropout": 0.1,
        },
    }


def _xattn_encoder_block(
    post_fusion_mode: str,
    era5_path: list[dict] | None = None,
    era5_output_channels: int = 128,
) -> dict:
    """Build the CrossAttentionFusionExtractor encoder block.

    Args:
        post_fusion_mode: "none", "ffn", or "self_attn_ffn".
        era5_path: override for the context ERA5 path. Defaults to base TCN path.
        era5_output_channels: output channels for the ERA5 context path (128 for TCN, 256 for XFMR).
    """
    base_init = cast(
        dict[str, Any],
        cast(
            list[dict[str, Any]],
            cast(dict[str, Any], BASE_CONFIG["model"])["init_args"]["model"][
                "init_args"
            ]["encoder"],
        )[0]["init_args"],
    )
    primary_path = copy.deepcopy(base_init["primary_path"])
    context_paths = copy.deepcopy(base_init["context_paths"])
    if era5_path is not None:
        context_paths[0] = era5_path
    return {
        "class_path": "rslearn.models.xatt_fusion.CrossAttentionFusionExtractor",
        "init_args": {
            "primary_path": primary_path,
            "context_paths": context_paths,
            "primary_output_channels": 768,
            "context_output_channels": [era5_output_channels],
            "attention_dim": 256,
            "num_memory_tokens": 4,
            "num_heads": 4,
            "attention_dropout": 0.1,
            "residual_dropout": 0.0,
            "post_fusion_mode": post_fusion_mode,
            "ffn_expansion": 2.0,
            "ffn_activation": "swiglu",
            "ffn_dropout": 0.1,
            "pre_fusion_dropout": 0.1,
        },
    }


# OlmoEarth (S2) primary and ERA5 context backbones under CrossAttentionFusionExtractor.
_XATTN_S2_PATH = "model.encoder.0.paths.0.0"
_XATTN_ERA5_PATH = "model.encoder.0.paths.1.0"

# Multiply all optimizer LRs at each encoder unfreeze (shared LR across the network).
LR_ON_UNFREEZE_DIV5 = 0.2  # divide LR by 5
LR_ON_UNFREEZE_DIV3_333 = 1 / 3  # divide LR by ~3


def _xattn_stages(
    s2_unfreeze_epoch: int = 10,
    scale_existing_groups: float | None = 0.333,
    era5_unfreeze_epoch: int | None = None,
    extra_frozen_selectors: list[str] | None = None,
    scale_only_on_first_unfreeze: bool = False,
) -> list[dict]:
    """Build fine-tuning stages for delayed S2 and optional delayed ERA5.

    When ``era5_unfreeze_epoch`` is set, both encoders start frozen at epoch 0.
    Unfreeze stages are ordered by epoch; whichever encoder unfreezes first keeps
    the other frozen via ``freeze_selectors`` so schedules like ERA5@10 / S2@20
    work correctly.

    ``scale_existing_groups`` is applied at **each** unfreeze (including the first
    of the two encoders): it multiplies every existing optimizer param group LR
    (`MultiStageFineTuning`), matching a shared single LR with ``unfreeze_lr_factor``
    1.0 for newly added groups.
    """
    s2 = _XATTN_S2_PATH
    era5 = _XATTN_ERA5_PATH

    extra_frozen_selectors = list(extra_frozen_selectors or [])
    if era5_unfreeze_epoch is not None and not extra_frozen_selectors:
        extra_frozen_selectors = [era5]

    freeze_selectors = [s2] + extra_frozen_selectors

    stages: list[dict] = [
        {
            "at_epoch": 0,
            "freeze_selectors": freeze_selectors,
            "unfreeze_selectors": [],
        },
    ]

    if era5_unfreeze_epoch is None:
        s2_stage: dict = {
            "at_epoch": s2_unfreeze_epoch,
            "freeze_selectors": [],
            "unfreeze_selectors": [s2],
            "unfreeze_lr_factor": 1.0,
        }
        if scale_existing_groups is not None:
            s2_stage["scale_existing_groups"] = scale_existing_groups
        stages.append(s2_stage)
        return stages

    def _unfreeze_stage(
        at_epoch: int,
        unfreeze: list[str],
        freeze_other: list[str],
        scale: float | None = scale_existing_groups,
    ) -> dict:
        st: dict = {
            "at_epoch": at_epoch,
            "freeze_selectors": freeze_other,
            "unfreeze_selectors": unfreeze,
            "unfreeze_lr_factor": 1.0,
        }
        if scale is not None:
            st["scale_existing_groups"] = scale
        return st

    if era5_unfreeze_epoch == s2_unfreeze_epoch:
        stages.append(
            _unfreeze_stage(
                era5_unfreeze_epoch,
                [s2, era5],
                [],
            )
        )
        return stages

    if era5_unfreeze_epoch < s2_unfreeze_epoch:
        stages.append(
            _unfreeze_stage(
                era5_unfreeze_epoch,
                [era5],
                [s2],
            )
        )
        stages.append(
            _unfreeze_stage(
                s2_unfreeze_epoch,
                [s2],
                [],
            )
        )
    else:
        stages.append(
            _unfreeze_stage(
                s2_unfreeze_epoch,
                [s2],
                [era5],
            )
        )
        stages.append(
            _unfreeze_stage(
                era5_unfreeze_epoch,
                [era5],
                [],
                None if scale_only_on_first_unfreeze else scale_existing_groups,
            )
        )

    return stages


def _logitres_era5_path(output_spatial_size: int = 10) -> list[dict]:
    """Build the ERA5 branch used by logit-residual fusion."""
    return [
        {
            "class_path": "rslearn.models.tcn_encoder.TCNEncoder",
            "init_args": {
                "in_channels": 14,
                "d_model": 128,
                "d_output": 128,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "pooling_windows": [1, 2, 4, 12],
                "dropout": 0.1,
                "output_spatial_size": output_spatial_size,
            },
        },
        {
            "class_path": "rslearn.models.conv.Conv",
            "init_args": {
                "in_channels": 128,
                "out_channels": 768,
                "kernel_size": 1,
                "padding": "same",
                "activation": {
                    "class_path": "torch.nn.Identity",
                },
            },
        },
    ]


# ── Commented baselines (full fusion width: S2=768, ERA5 XFMR=256; no _b* bottleneck)
# Still valid: ``_xattn_encoder_block_xfmr(..., fusion_bottleneck_dim=None)`` is the default.
# ─────────────────────────────────────────────────────────────────────────────
#     make_experiment(
#         "xattn_selfattnffn_shallow_lr1e6_era5mask30_unf15_div10_unf15",
#         {
#             "model.init_args.model.init_args.encoder.0": _xattn_encoder_block(
#                 post_fusion_mode="self_attn_ffn"
#             ),
#             "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#                 in_channels=768
#             ),
#             "data.init_args.train_config.transforms": _train_transforms(
#                 era5_mask_ratio=0.30
#             ),
#             "trainer.callbacks.3.init_args.stages": _xattn_stages(
#                 s2_unfreeze_epoch=15,
#                 scale_existing_groups=0.1,
#             ),
#         },
#     ),
#

BEST_XATTN_XFMR_BASE_NAME = "base"
_BEST_XATTN_INIT = "model.init_args.model.init_args.encoder.0.init_args"
_BEST_SEG_HEAD_INIT = (
    "model.init_args.model.init_args.decoders.segmentation.1.init_args"
)
_BEST_STAGES = "trainer.callbacks.3.init_args.stages"
_BEST_RESTORE_CONFIG = "model.init_args.restore_config"


def make_best_xattn_xfmr_experiment(
    suffix: str = "",
    overrides: dict[str, Any] | None = None,
) -> tuple[str, dict]:
    """Create an experiment by applying overrides on top of the best current run."""
    merged_overrides = _best_xattn_xfmr_base_overrides()
    if overrides:
        merged_overrides.update(overrides)
    name = (
        BEST_XATTN_XFMR_BASE_NAME
        if not suffix
        else f"{BEST_XATTN_XFMR_BASE_NAME}_{suffix}"
    )
    return make_experiment(name, merged_overrides)


def _best_xattn_xfmr_base_overrides() -> dict[str, Any]:
    """Return the current best late-fusion config as reusable overrides."""
    return {
        "model.init_args.model.init_args.encoder.0": _xattn_encoder_block_xfmr(
            post_fusion_mode="self_attn_ffn"
        ),
        "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
            in_channels=768
        ),
        "data.init_args.train_config.transforms": _train_transforms(
            era5_mask_ratio=0.40
        ),
        _BEST_STAGES: _xattn_stages(
            s2_unfreeze_epoch=15,
            scale_existing_groups=0.1,
        ),
    }


EXPERIMENTS = [
    # make_experiment(
    #     "base_rc",
    #     {
    #         "model.init_args.model.init_args.encoder.0": _xattn_encoder_block_xfmr(
    #             post_fusion_mode="self_attn_ffn"
    #         ),
    #         "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
    #             in_channels=768
    #         ),
    #         "data.init_args.train_config.transforms": _train_transforms(
    #             era5_mask_ratio=0.40
    #         ),
    #         "trainer.callbacks.3.init_args.stages": _xattn_stages(
    #             s2_unfreeze_epoch=15,
    #             scale_existing_groups=0.1,
    #         ),
    #         "model.init_args.optimizer.init_args.lr": 1e-6,
    #     },
    # ),
    # make_experiment(
    #     "base_lr5e6_rc",
    #     {
    #         "model.init_args.model.init_args.encoder.0": _xattn_encoder_block_xfmr(
    #             post_fusion_mode="self_attn_ffn"
    #         ),
    #         "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
    #             in_channels=768
    #         ),
    #         "data.init_args.train_config.transforms": _train_transforms(
    #             era5_mask_ratio=0.40
    #         ),
    #         "trainer.callbacks.3.init_args.stages": _xattn_stages(
    #             s2_unfreeze_epoch=15,
    #             scale_existing_groups=0.1,
    #         ),
    #         "model.init_args.optimizer.init_args.lr": 5e-6,
    #     },
    # ),
    # make_best_xattn_xfmr_experiment(
    #     "b2_focal_pos_a08_g20_rc",
    #     {
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.8,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 2.0,
    #         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
    #         "model.init_args.optimizer.init_args.lr": 1e-6,
    #     },
    # ),
    # make_best_xattn_xfmr_experiment(
    #     "b3_focal_pos_a09_g30_rc",
    #     {
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.9,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 3.0,
    #         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
    #         "model.init_args.optimizer.init_args.lr": 1e-6,
    #     },
    # ),
    # make_best_xattn_xfmr_experiment(
    #     "b3_focal_pos_a09_g30_rc_v3",
    #     {
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.9,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 3.0,
    #         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
    #         "model.init_args.optimizer.init_args.lr": 1e-6,
    #     },
    # ),
    make_best_xattn_xfmr_experiment(
        "b3_focal_pos_a09_g30_rc_era5warm_lr1e6",
        {
            f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
            f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.9,
            f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 3.0,
            "model.init_args.optimizer.init_args.lr": 1e-6,
            _BEST_RESTORE_CONFIG: _era5_xfmr_warm_restore_config(),
            _BEST_STAGES: _xattn_stages(
                s2_unfreeze_epoch=15,
                era5_unfreeze_epoch=20,
                scale_existing_groups=0.1,
                scale_only_on_first_unfreeze=True,
            ),
        },
    ),
    make_best_xattn_xfmr_experiment(
        "b3_focal_pos_a09_g30_rc_era5warm_lr1e6_ep60",
        {
            f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
            f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.9,
            f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 3.0,
            "model.init_args.optimizer.init_args.lr": 1e-6,
            "trainer.max_epochs": 60,
            "model.init_args.scheduler.init_args.T_max": 60,
            _BEST_RESTORE_CONFIG: _era5_xfmr_warm_restore_config(),
            _BEST_STAGES: _xattn_stages(
                s2_unfreeze_epoch=15,
                era5_unfreeze_epoch=25,
                scale_existing_groups=0.1,
                scale_only_on_first_unfreeze=True,
            ),
        },
    ),
    # make_best_xattn_xfmr_experiment(
    #     "b3_focal_pos_a09_g30_rc_ep80",
    #     {
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.9,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 3.0,
    #         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
    #         "model.init_args.optimizer.init_args.lr": 1e-6,
    #         "trainer.max_epochs": 80,
    #         "model.init_args.scheduler.init_args.T_max": 80,
    #     },
    # ),
    # make_best_xattn_xfmr_experiment(
    #     "b3_focal_pos_a09_g30_lr5e6_rc",
    #     {
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.9,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 3.0,
    #         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
    #         "model.init_args.optimizer.init_args.lr": 5e-6,
    #     },
    # ),
    # make_best_xattn_xfmr_experiment(
    #     "b3_focal_pos_a09_g30_lr1e5_rc",
    #     {
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.9,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 3.0,
    #         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
    #         "model.init_args.optimizer.init_args.lr": 1e-5,
    #     },
    # ),
    # make_best_xattn_xfmr_experiment(
    #     "b4_focal_pos_a095_g40",
    #     {
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.95,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 4.0,
    #         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
    #     },
    # ),
    # make_best_xattn_xfmr_experiment(
    #     "b4_focal_pos_a095_g50_rc",
    #     {
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.95,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 5.0,
    #         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
    #     },
    # ),
    # make_best_xattn_xfmr_experiment(
    #     "b4_focal_pos_a09_g40",
    #     {
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.9,
    #         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 4.0,
    #         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
    #     },
    # ),
]


# E4 is intentionally omitted as a separate run because the current best config
# already uses ``pre_fusion_dropout=0.1`` via ``_xattn_encoder_block_xfmr``.
# EXPERIMENTS = [
# make_best_xattn_xfmr_experiment(),
# # A family: cls-loss reduction
# make_best_xattn_xfmr_experiment(
#     "a1_cls05_dice15",
#     {
#         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [0.5, 1.5],
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "a2_cls067_dice133",
#     {
#         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [2 / 3, 4 / 3],
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "a3_diceonly",
#     {
#         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [0.0, 1.0],
#     },
# ),
# # B family: focal loss
# make_best_xattn_xfmr_experiment(
#     "b1_focal",
#     {
#         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
#         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "b2_focal_pos_a08_g20",
#     {
#         f"{_BEST_SEG_HEAD_INIT}.focal_loss": True,
#         f"{_BEST_SEG_HEAD_INIT}.focal_loss_alpha": 0.8,
#         f"{_BEST_SEG_HEAD_INIT}.focal_loss_gamma": 2.0,
#         f"{_BEST_SEG_HEAD_INIT}.loss_weights": [1.0, 1.0],
#     },
# ),
# C family: auxiliary S2 loss
# make_best_xattn_xfmr_experiment(
#     "c1_s2aux01",
#     {
#         "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#             in_channels=768,
#             context_key="path0_intermediate",
#         ),
#         f"{_BEST_SEG_HEAD_INIT}.path0_aux_weight": 0.1,
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "c2_s2aux025",
#     {
#         "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#             in_channels=768,
#             context_key="path0_intermediate",
#         ),
#         f"{_BEST_SEG_HEAD_INIT}.path0_aux_weight": 0.25,
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "c3_s2aux025_era5warm_e5f20",
#     {
#         "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#             in_channels=768,
#             context_key="path0_intermediate",
#         ),
#         f"{_BEST_SEG_HEAD_INIT}.path0_aux_weight": 0.25,
#         _BEST_RESTORE_CONFIG: _era5_xfmr_warm_restore_config(),
#         _BEST_STAGES: _xattn_stages(
#             s2_unfreeze_epoch=15,
#             era5_unfreeze_epoch=20,
#             scale_existing_groups=0.1,
#         ),
#     },
# ),
# # D family: ERA5 branch dropout
# make_best_xattn_xfmr_experiment(
#     "d1_pathdrop015",
#     {
#         f"{_BEST_XATTN_INIT}.context_dropout_prob": 0.15,
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "d2_pathdrop03",
#     {
#         f"{_BEST_XATTN_INIT}.context_dropout_prob": 0.3,
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "d3_pathdrop03_era5warm_e5f20",
#     {
#         f"{_BEST_XATTN_INIT}.context_dropout_prob": 0.3,
#         _BEST_RESTORE_CONFIG: _era5_xfmr_warm_restore_config(),
#         _BEST_STAGES: _xattn_stages(
#             s2_unfreeze_epoch=15,
#             era5_unfreeze_epoch=20,
#             scale_existing_groups=0.1,
#         ),
#     },
# ),
# # E family: x-attention overfitting reduction
# make_best_xattn_xfmr_experiment(
#     "e1_attnd128",
#     {
#         f"{_BEST_XATTN_INIT}.attention_dim": 128,
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "e2_attnd64",
#     {
#         f"{_BEST_XATTN_INIT}.attention_dim": 64,
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "e3_attnd128_mem2",
#     {
#         f"{_BEST_XATTN_INIT}.attention_dim": 128,
#         f"{_BEST_XATTN_INIT}.num_memory_tokens": 2,
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "e4_pfdrop02",
#     {
#         f"{_BEST_XATTN_INIT}.pre_fusion_dropout": 0.2,
#     },
# ),
# make_best_xattn_xfmr_experiment(
#     "e5_pfdrop02",
#     {
#         f"{_BEST_XATTN_INIT}.pre_fusion_dropout": 0.3,
#     },
# ),
# ]


#     # Same as commented baseline ``era5xfmr_era5mask30_unf15_div10`` (S2-only freeze,
#     # unfreeze @15, scale 0.1) + fusion bottleneck D. No ERA5 warm restore / no ERA5 freeze.
#     make_experiment(
#         "xattn_selfattnffn_shallow_lr1e6_era5xfmr_era5mask30_unf15_div10_b32",
#         {
#             "model.init_args.model.init_args.encoder.0": _xattn_encoder_block_xfmr(
#                 "self_attn_ffn",
#                 fusion_bottleneck_dim=32,
#             ),
#             "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#                 in_channels=32
#             ),
#             "data.init_args.train_config.transforms": _train_transforms(
#                 era5_mask_ratio=0.30
#             ),
#             "trainer.callbacks.3.init_args.stages": _xattn_stages(
#                 s2_unfreeze_epoch=15,
#                 scale_existing_groups=0.1,
#             ),
#         },
#     ),
#     make_experiment(
#         "xattn_selfattnffn_shallow_lr1e6_era5xfmr_era5mask30_unf15_div10_b64",
#         {
#             "model.init_args.model.init_args.encoder.0": _xattn_encoder_block_xfmr(
#                 "self_attn_ffn",
#                 fusion_bottleneck_dim=64,
#             ),
#             "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#                 in_channels=64
#             ),
#             "data.init_args.train_config.transforms": _train_transforms(
#                 era5_mask_ratio=0.30
#             ),
#             "trainer.callbacks.3.init_args.stages": _xattn_stages(
#                 s2_unfreeze_epoch=15,
#                 scale_existing_groups=0.1,
#             ),
#         },
#     ),
#     make_experiment(
#         "xattn_selfattnffn_shallow_lr1e6_era5xfmr_era5mask30_unf15_div10_b128",
#         {
#             "model.init_args.model.init_args.encoder.0": _xattn_encoder_block_xfmr(
#                 "self_attn_ffn",
#                 fusion_bottleneck_dim=128,
#             ),
#             "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#                 in_channels=128
#             ),
#             "data.init_args.train_config.transforms": _train_transforms(
#                 era5_mask_ratio=0.30
#             ),
#             "trainer.callbacks.3.init_args.stages": _xattn_stages(
#                 s2_unfreeze_epoch=15,
#                 scale_existing_groups=0.1,
#             ),
#         },
#     ),
#     # Same as above + ERA5 XFMR warm-start checkpoint (``_era5_xfmr_warm_restore_config``).
#     make_experiment(
#         "xattn_selfattnffn_shallow_lr1e6_era5xfmrwarm_era5mask30_unf15_div10_b32",
#         {
#             "model.init_args.model.init_args.encoder.0": _xattn_encoder_block_xfmr(
#                 "self_attn_ffn",
#                 fusion_bottleneck_dim=32,
#             ),
#             "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#                 in_channels=32
#             ),
#             "model.init_args.restore_config": _era5_xfmr_warm_restore_config(),
#             "data.init_args.train_config.transforms": _train_transforms(
#                 era5_mask_ratio=0.30
#             ),
#             "trainer.callbacks.3.init_args.stages": _xattn_stages(
#                 s2_unfreeze_epoch=15,
#                 scale_existing_groups=0.1,
#             ),
#         },
#     ),
#     make_experiment(
#         "xattn_selfattnffn_shallow_lr1e6_era5xfmrwarm_era5mask30_unf15_div10_b64",
#         {
#             "model.init_args.model.init_args.encoder.0": _xattn_encoder_block_xfmr(
#                 "self_attn_ffn",
#                 fusion_bottleneck_dim=64,
#             ),
#             "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#                 in_channels=64
#             ),
#             "model.init_args.restore_config": _era5_xfmr_warm_restore_config(),
#             "data.init_args.train_config.transforms": _train_transforms(
#                 era5_mask_ratio=0.30
#             ),
#             "trainer.callbacks.3.init_args.stages": _xattn_stages(
#                 s2_unfreeze_epoch=15,
#                 scale_existing_groups=0.1,
#             ),
#         },
#     ),
#     make_experiment(
#         "xattn_selfattnffn_shallow_lr1e6_era5xfmrwarm_era5mask30_unf15_div10_b128",
#         {
#             "model.init_args.model.init_args.encoder.0": _xattn_encoder_block_xfmr(
#                 "self_attn_ffn",
#                 fusion_bottleneck_dim=128,
#             ),
#             "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#                 in_channels=128
#             ),
#             "model.init_args.restore_config": _era5_xfmr_warm_restore_config(),
#             "data.init_args.train_config.transforms": _train_transforms(
#                 era5_mask_ratio=0.30
#             ),
#             "trainer.callbacks.3.init_args.stages": _xattn_stages(
#                 s2_unfreeze_epoch=15,
#                 scale_existing_groups=0.1,
#             ),
#         },
#     ),
# ]

# ── Legacy reference run (used old LateFusionFeatureExtractor API — not compatible
# with xatt_fusion.CrossAttentionFusionExtractor).
# Path 1 is projected 128 → 768 via a 1×1 Conv so channels match path 0.
# No auxiliary loss, no modality dropout, shallow decoder.
#     make_experiment("logitres_a01_lr1e6_era5warm_v2", {
#         "model.init_args.model.init_args.encoder.0.init_args.fusion_mode": "logit_residual",  # OLD API
#         "model.init_args.model.init_args.encoder.0.init_args.logit_residual_alpha": 0.1,      # OLD API
#         "model.init_args.model.init_args.encoder.0.init_args.context_paths.0": _logitres_era5_path(),  # OLD API key path
#         "model.init_args.model.init_args.decoders.segmentation": _shallow_decoder(
#             in_channels=768
#         ),
#         "model.init_args.restore_config": _era5_warm_restore_config(),
#         "data.init_args.train_config.transforms": _train_transforms(
#                 era5_mask_ratio=0.30
#             ),
#             "trainer.callbacks.3.init_args.stages": _xattn_stages(
#                 s2_unfreeze_epoch=15,
#                 scale_existing_groups=0.1,
#             ),
#         },
#     ),


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
    "latefusion",
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
    opt = config["model"]["init_args"]["optimizer"]["init_args"]
    encoder_block = config["model"]["init_args"]["model"]["init_args"]["encoder"][0]
    encoder_args = encoder_block["init_args"]

    post_mode = encoder_args.get("post_fusion_mode", "none")
    fusion_mode = f"xattn/{post_mode}"
    attn_dim = encoder_args.get("attention_dim")
    # ERA5 context path (context_paths[0], module 0) — TCN or Transformer
    era5_mod = encoder_args["context_paths"][0][0]
    era5_args = era5_mod["init_args"]
    era5_class = era5_mod.get("class_path", "")
    if "PatchTransformerEncoder" in era5_class:
        era5_label = f"xfmr_d{era5_args['d_model']}_do{era5_args['dropout']}"
    else:
        era5_label = f"tcn_d={era5_args['d_model']}, tcn_drop={era5_args['dropout']}"
    # Find the SegmentationHead and first Conv in the decoder list
    seg_decoder = config["model"]["init_args"]["model"]["init_args"]["decoders"][
        "segmentation"
    ]
    seg_head: dict = {}
    conv_in = "?"
    decoder_depth = 0
    for module in seg_decoder:
        cp = module.get("class_path", "")
        if "SegmentationHead" in cp:
            seg_head = module.get("init_args", {})
        if "Conv" in cp:
            decoder_depth += 1
            if conv_in == "?":
                conv_in = module["init_args"]["in_channels"]
    bs = config["data"]["init_args"]["batch_size"]
    epochs = config["trainer"]["max_epochs"]
    weights = seg_head.get("weights")
    dice = seg_head.get("dice_loss", False)
    loss_weights = seg_head.get("loss_weights", [1.0, 1.0])
    focal_loss = seg_head.get("focal_loss", False)
    focal_alpha = seg_head.get("focal_loss_alpha", 0.25)
    focal_gamma = seg_head.get("focal_loss_gamma", 2.0)
    path0_aux_weight = seg_head.get("path0_aux_weight", 0.0)
    # Optimizer details — handle both AdamW and GroupAdamW
    if "lr" in opt:
        lr_str = f"lr={opt['lr']}, wd={opt['weight_decay']}"
    else:
        # GroupAdamW: show per-group LRs
        groups_str = ", ".join(
            f"{g['prefix'].rstrip('.')}={g['lr']}" for g in opt.get("groups", [])
        )
        lr_str = f"groups=[{groups_str}], default_lr={opt['default_lr']}, wd={opt['default_weight_decay']}"
    # Fusion details
    fusion_extra = (
        f", attn_dim={attn_dim}, mem={encoder_args.get('num_memory_tokens', '?')}, "
        f"attn_drop={encoder_args.get('attention_dropout', 0.0)}, "
        f"res_drop={encoder_args.get('residual_dropout', 0.0)}, "
        f"ffn_drop={encoder_args.get('ffn_dropout', 0.0)}, "
        f"pre_fusion_drop={encoder_args.get('pre_fusion_dropout', 0.0)}, "
        f"ctx_drop={encoder_args.get('context_dropout_prob', 0.0)}"
    )
    decoder_label = f"deep({decoder_depth}L)" if decoder_depth > 1 else "shallow"
    cls_label = f"focal(a={focal_alpha},g={focal_gamma})" if focal_loss else "xent"
    return (
        f"  fusion={fusion_mode}{fusion_extra}, decoder={decoder_label}, conv_in={conv_in}, "
        f"era5={era5_label}, "
        f"{lr_str}, weights={weights}, cls={cls_label}, dice={dice}, "
        f"loss_w={loss_weights}, aux_w={path0_aux_weight}, "
        f"bs={bs}, epochs={epochs}"
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
    """Generate configs and launch the requested Canada NBAC late-fusion runs."""
    parser = argparse.ArgumentParser(
        description="Launch CanadaFireSat late-fusion experiments"
    )
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
# python -m rslp.wildfire.Canada_nbac.launch_experiments_latefusion --dry-run

# Launch everything:
# python -m rslp.wildfire.Canada_nbac.launch_experiments_latefusion

# Launch only specific experiments:
# python -m rslp.wildfire.Canada_nbac.launch_experiments_latefusion --only lf_s2era5_baseline

# Run locally (test dataset, no wandb):
# python -m rslp.wildfire.Canada_nbac.launch_experiments_latefusion --local
# python -m rslp.wildfire.Canada_nbac.launch_experiments_latefusion --local --only lf_s2era5_baseline --dry-run


# Example:
# cd /weka/dfive-default/hadriens/rslearn_projects
# source ./.venv/bin/activate
# python -m rslp.wildfire.Canada_nbac.launch_experiments_latefusion
