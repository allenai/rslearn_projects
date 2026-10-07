"""Write the model configs for the change alert experiment grid.

Each run is written as a complete YAML file (configs/{source}/{run}.yaml) rather than
being composed from several --config files, since composing configs replaces lists
such as the inputs' layers and the transforms wholesale.

Runs differ along three axes:
- aug: fixed4 trains on the four fixed slots (7/35/62/90 days after the change),
  rand4 on the four per-window randomized slots, and rand1 on randomized slot 0 only.
- ctx: history uses 8 quarterly + 4 weekly images, recent uses 12 weekly images.
- head: pool predicts the category from attention-pooled tokens, breakpoint from the
  before/after features of a BreakpointScan.

All runs validate on randomized slot 0 so that validation covers the whole detection
range and is identical across runs. make_test_config derives the config for one test
scenario from a training config; eval_test.py uses it.
"""

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from common import (
    HISTORY_LOOKBACK,
    NUM_SLOTS,
    SOURCE_DATASETS,
    dataset_path,
)

PROJECT_NAME = "2026_10_07_change_alerts"
CONFIG_DIR = Path(__file__).parent / "configs"

# OlmoEarth band order.
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
EMBED_DIM = 768
NUM_TIMESTEPS = 12
VAL_SLOT = "rand_0"


@dataclass(frozen=True)
class Run:
    """One configuration in the experiment grid."""

    aug: str
    ctx: str
    head: str

    @property
    def name(self) -> str:
        """The run name, also used as the config filename."""
        return f"{self.aug}_{self.ctx}_{self.head}"

    @property
    def train_slots(self) -> list[str]:
        """The slots to sample the training time series from."""
        if self.aug == "fixed4":
            return [f"fixed_{k}" for k in range(NUM_SLOTS)]
        if self.aug == "rand4":
            return [f"rand_{k}" for k in range(NUM_SLOTS)]
        if self.aug == "rand1":
            return ["rand_0"]
        raise ValueError(f"unknown aug {self.aug}")


RUNS = [
    Run("fixed4", "history", "pool"),
    Run("rand4", "history", "pool"),
    Run("rand1", "history", "pool"),
    Run("fixed4", "recent", "pool"),
    Run("fixed4", "history", "breakpoint"),
    Run("fixed4", "recent", "breakpoint"),
]
RUNS_BY_NAME = {run.name: run for run in RUNS}


def _image_input(layer: str) -> dict[str, Any]:
    return {
        "data_type": "raster",
        "layers": [layer],
        "bands": S2_BANDS,
        "passthrough": True,
        "dtype": "FLOAT32",
        "load_all_layers": True,
        "load_all_item_groups": True,
    }


def _slot_inputs(ctx: str, slot: str) -> dict[str, dict[str, Any]]:
    """Get the inputs (keyed by layer name) for one slot."""
    layers = [f"freq_{slot}"]
    if ctx == "history":
        layers.append(f"infreq_{slot}")
    return {layer: _image_input(layer) for layer in layers}


def _option(ctx: str, slot: str) -> dict[str, str]:
    option = {"frequent": f"freq_{slot}"}
    if ctx == "history":
        option["infrequent"] = f"infreq_{slot}"
    return option


def _sampler(
    ctx: str, slots: list[str], option_index: int | None, drop_keys: list[str]
) -> dict[str, Any]:
    init_args: dict[str, Any] = {
        "options": [_option(ctx, slot) for slot in slots],
        "output_key": "sentinel2_l2a",
        "category_target": "category",
        "timestep_target": "timestep",
    }
    if ctx == "history":
        init_args.update(
            num_frequent=4,
            frequent_lookback_days=HISTORY_LOOKBACK.days,
            num_infrequent=8,
        )
    elif ctx == "recent":
        init_args.update(num_frequent=NUM_TIMESTEPS)
    else:
        raise ValueError(f"unknown ctx {ctx}")
    if option_index is not None:
        init_args["option_index"] = option_index
    if drop_keys:
        init_args["drop_keys"] = drop_keys
    return {
        "class_path": "rslearn.change_alerts.sampler.ChangeTimeSeriesSampler",
        "init_args": init_args,
    }


NORMALIZE = {
    "class_path": "rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize",
    "init_args": {"band_names": {"sentinel2_l2a": S2_BANDS}},
}

FLIP = {
    "class_path": "rslearn.train.transforms.flip.Flip",
    "init_args": {
        "image_selectors": [
            "sentinel2_l2a",
            "target/category/classes",
            "target/category/valid",
            "target/timestep/classes",
            "target/timestep/valid",
        ]
    },
}


def _conv(in_channels: int, out_channels: int, kernel_size: int = 3) -> dict:
    return {
        "class_path": "rslearn.models.conv.Conv",
        "init_args": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
        },
    }


def _upsample(scale_factor: int) -> dict:
    return {
        "class_path": "rslearn.models.upsample.Upsample",
        "init_args": {"scale_factor": scale_factor},
    }


def _category_decoder(head: str, num_classes: int) -> list[dict[str, Any]]:
    if head == "pool":
        first: dict[str, Any] = {
            "class_path": "rslearn.models.attention_pooling.SimpleAttentionPool",
            "init_args": {"in_dim": EMBED_DIM},
        }
        first_channels = EMBED_DIM
    elif head == "breakpoint":
        first = {
            "class_path": "rslearn.models.breakpoint_scan.BreakpointScan",
            "init_args": {"in_dim": EMBED_DIM, "output": "BEFORE_AFTER", "hidden": 256},
        }
        first_channels = 2 * EMBED_DIM
    else:
        raise ValueError(f"unknown head {head}")
    classifier = _conv(64, num_classes, kernel_size=1)
    classifier["init_args"]["activation"] = {"class_path": "torch.nn.Identity"}
    return [
        first,
        _conv(first_channels, 256),
        _upsample(2),
        _conv(256, 128),
        _upsample(2),
        _conv(128, 64),
        classifier,
        {"class_path": "rslearn.train.tasks.segmentation.SegmentationHead"},
    ]


TIMESTEP_DECODER = [
    {
        "class_path": "rslearn.models.tokens_to_channels.TokensToChannels",
        # Padded timesteps (when a batch has shorter time series) are never predicted.
        "init_args": {"in_dim": EMBED_DIM, "out_dim": 1, "mask_fill_value": -10000.0},
    },
    _upsample(4),
    {
        "class_path": "rslearn.train.tasks.per_pixel_timestep.PerPixelTimestepHead",
        "init_args": {"input_key": "sentinel2_l2a"},
    },
]


def _task(categories: list[str]) -> dict[str, Any]:
    num_classes = len(categories) + 1
    return {
        "class_path": "rslearn.train.tasks.multi_task.MultiTask",
        "init_args": {
            "input_mapping": {
                "category": {"label_category": "targets"},
                # Placeholder: ChangeTimeSeriesSampler overwrites the timestep target.
                "timestep": {"change_day": "targets"},
            },
            "tasks": {
                "category": {
                    "class_path": "rslearn.train.tasks.segmentation.SegmentationTask",
                    "init_args": {
                        "num_classes": num_classes,
                        "nodata_value": 0,
                        "class_names": ["nodata"] + categories,
                        "metric_kwargs": {"average": "micro"},
                        "other_metrics": {
                            "balanced_accuracy": {
                                "class_path": "rslearn.change_alerts.metrics.BalancedAccuracy",
                                "init_args": {"num_classes": num_classes},
                            },
                            "change_auroc": {
                                "class_path": "rslearn.change_alerts.metrics.ChangeAUROC",
                                "init_args": {"none_class": 1},
                            },
                        },
                    },
                },
                "timestep": {
                    "class_path": "rslearn.train.tasks.per_pixel_timestep.PerPixelTimestepTask",
                    "init_args": {
                        "num_classes": NUM_TIMESTEPS,
                        # The built-in accuracy requires exactly num_classes channels,
                        # which batches of shorter time series do not have.
                        "enable_accuracy_metric": False,
                        "other_metrics": {
                            "accuracy": {
                                "class_path": "rslearn.change_alerts.metrics.TimestepToleranceAccuracy",
                                "init_args": {"tolerance": 0},
                            },
                            "within1": {
                                "class_path": "rslearn.change_alerts.metrics.TimestepToleranceAccuracy",
                                "init_args": {"tolerance": 1},
                            },
                        },
                    },
                },
            },
        },
    }


def make_train_config(source: str, run: Run, ds_path: str | None = None) -> dict:
    """Make the training config for one source dataset and run."""
    categories = SOURCE_DATASETS[source].categories
    slots = list(run.train_slots)
    if VAL_SLOT not in slots:
        slots.append(VAL_SLOT)

    inputs: dict[str, Any] = {}
    for slot in slots:
        inputs.update(_slot_inputs(run.ctx, slot))
    inputs["label_category"] = {
        "data_type": "raster",
        "layers": ["label_category"],
        "bands": ["label"],
        "is_target": True,
        "dtype": "INT32",
    }
    # The sampler reads the change day from the input dict to derive the targets.
    inputs["change_day"] = {
        "data_type": "raster",
        "layers": ["label_change_day"],
        "bands": ["label"],
        "is_target": True,
        "passthrough": True,
        "dtype": "INT32",
    }

    val_inputs = list(_slot_inputs(run.ctx, VAL_SLOT).keys())
    train_drop = [key for key in val_inputs if VAL_SLOT not in run.train_slots]
    val_drop = [
        key
        for slot in run.train_slots
        if slot != VAL_SLOT
        for key in _slot_inputs(run.ctx, slot)
    ]

    return {
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
                                    "model_id": "OLMOEARTH_V1_2_BASE",
                                    "patch_size": 4,
                                    "token_pooling": False,
                                    "use_legacy_timestamps": False,
                                },
                            }
                        ],
                        "decoders": {
                            "category": _category_decoder(
                                run.head, len(categories) + 1
                            ),
                            "timestep": TIMESTEP_DECODER,
                        },
                    },
                },
                "optimizer": {
                    "class_path": "rslearn.models.olmoearth_pretrain.optimizer.LayerDecayAdamW",
                    "init_args": {
                        "lr": 0.0001,
                        "layer_decay_rate": 0.65,
                        "num_layers": 12,
                        "encoder_prefix": "model.encoder.0",
                    },
                },
                "scheduler": {
                    "class_path": "rslearn.train.scheduler.PlateauScheduler",
                    "init_args": {
                        "factor": 0.2,
                        "patience": 5,
                        "min_lr": 0,
                        "cooldown": 10,
                    },
                },
            },
        },
        "data": {
            "class_path": "rslearn.train.data_module.RslearnDataModule",
            "init_args": {
                "path": ds_path or dataset_path(source, "train"),
                "inputs": inputs,
                "task": _task(categories),
                "batch_size": 8,
                "num_workers": 16,
                "train_config": {
                    "transforms": [
                        _sampler(run.ctx, run.train_slots, None, train_drop),
                        FLIP,
                        NORMALIZE,
                    ],
                    "tags": {"split": "train"},
                },
                "val_config": {
                    "transforms": [
                        _sampler(run.ctx, [VAL_SLOT], 0, val_drop),
                        NORMALIZE,
                    ],
                    "tags": {"split": "val"},
                },
                "test_config": {
                    "transforms": [
                        _sampler(run.ctx, [VAL_SLOT], 0, val_drop),
                        NORMALIZE,
                    ],
                    "tags": {"split": "val"},
                },
            },
        },
        "trainer": {
            "max_epochs": 100,
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {"logging_interval": "epoch"},
                },
                {
                    "class_path": "rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint",
                    "init_args": {
                        "monitor": "val_category/balanced_accuracy",
                        "mode": "max",
                    },
                },
            ],
        },
        "project_name": PROJECT_NAME,
        "run_name": f"{source}_{run.name}",
        "management_dir": "${RSLP_PREFIX}/projects",
    }


def make_test_config(
    train_config: dict, source: str, run: Run, end_days: int, ds_path: str | None = None
) -> dict:
    """Make the config to test a trained run on one test scenario.

    The project_name and run_name are kept so the best checkpoint of the run is
    loaded. The test datasets have one slot per scenario, freq_test_{E} (and
    infreq_test_{E} for the history context), with the time series ending E days
    after the change.
    """
    config = copy.deepcopy(train_config)
    init_args = config["data"]["init_args"]
    init_args["path"] = ds_path or dataset_path(source, f"test_{run.ctx}")
    slot = f"test_{end_days}"
    inputs = {
        key: value
        for key, value in init_args["inputs"].items()
        if key in ("label_category", "change_day")
    }
    inputs.update(_slot_inputs(run.ctx, slot))
    init_args["inputs"] = inputs
    init_args["test_config"] = {
        "transforms": [_sampler(run.ctx, [slot], 0, []), NORMALIZE],
        "tags": {"split": "test"},
    }
    # Only the test split is used, but the other splits must not reference inputs
    # that no longer exist.
    for split in ("train_config", "val_config"):
        init_args[split] = init_args["test_config"]
    return config


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def write_config(config: dict, fname: Path) -> None:
    """Write a config as YAML without anchors and aliases."""
    with fname.open("w") as f:
        yaml.dump(config, f, Dumper=_NoAliasDumper, sort_keys=False)


def main() -> None:
    """Write the training configs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", default=",".join(SOURCE_DATASETS), help="Comma-separated sources"
    )
    parser.add_argument("--out_dir", default=str(CONFIG_DIR))
    args = parser.parse_args()

    for source in args.sources.split(","):
        source_dir = Path(args.out_dir) / source
        source_dir.mkdir(parents=True, exist_ok=True)
        for run in RUNS:
            out_fname = source_dir / f"{run.name}.yaml"
            write_config(make_train_config(source, run), out_fname)
            print(f"wrote {out_fname}")


if __name__ == "__main__":
    main()
