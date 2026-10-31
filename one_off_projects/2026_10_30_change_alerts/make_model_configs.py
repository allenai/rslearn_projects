"""Write the model configs for the change alert runs (round 3).

Each run is written as a complete YAML file (configs/{source}/{run}.yaml) rather than
being composed from several --config files, since composing configs replaces lists
such as the inputs' layers and the transforms wholesale.

The model is the same as the round 2 `fixed4_history_pool` baseline
(2026_10_07_change_alerts): OlmoEarth v1.2 Base, an attention-pooled category head, and
a per-timestep change timestep head, on 8 quarterly + 4 weekly Sentinel-2 L2A mosaics.
What changed is the dataset and the sampler (rslearn.change_alerts.layers and
ChangeTimeSeriesSampler):

- Training: the sampler picks 4 consecutive weekly mosaics at a random position in the
  training window (so the change is between 1 and 13 weeks before the end of the time
  series), sometimes keeps only the latest 1-3 of them, and adds the 8 quarterly
  mosaics before them, with a stand-in weekly mosaic for the latest quarterly mosaic
  when the pick starts at least 45 days into the training window.
- Validation and test: prediction windows (the 4 weeks before E days after the change)
  with the sampler using the latest frequent images (select: latest). The val split
  has all scenarios E in END_DAYS; make_test_config selects one scenario of the test
  split.

The best checkpoint is selected by val_category/balanced_accuracy, as in round 2.

The runs are history_pool with seeds 0, 1, and 2 (rounds 1 and 2 implicitly used seed
0). make_old_run_config makes the config of a round 1 or 2 run (for testing its
checkpoint on the round 3 test scenarios).
"""

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from common import (
    INFREQUENT_PERIOD,
    NUM_INFREQUENT,
    NUM_TIMESTEPS,
    ROUND2_DATASET_ROOT,
    SOURCE_DATASETS,
    TIMING,
    dataset_path,
)

PROJECT_NAME = "2026_10_30_change_alerts"
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
# jsonargparse parses timedelta from str(timedelta).
STAND_IN_MIN_GAP = str(INFREQUENT_PERIOD / 2)


@dataclass(frozen=True)
class Run:
    """One run: the category head and the seed."""

    head: str = "pool"
    seed: int = 0

    @property
    def name(self) -> str:
        """The run name, also used as the config filename."""
        return f"history_{self.head}_s{self.seed}"


RUNS = [Run("pool", seed) for seed in range(3)]
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


LABEL_INPUTS = {
    "label_category": {
        "data_type": "raster",
        "layers": ["label_category"],
        "bands": ["label"],
        "is_target": True,
        "dtype": "INT32",
    },
    "change_day": {
        "data_type": "raster",
        "layers": ["label_change_day"],
        "bands": ["label"],
        "is_target": True,
        "dtype": "INT32",
    },
}


def _inputs(frequent_layer: str, infrequent_layer: str) -> dict[str, Any]:
    """Get the data inputs, with the frequent and infrequent inputs as freq/infreq."""
    return {
        "freq": _image_input(frequent_layer),
        "infreq": _image_input(infrequent_layer),
        **copy.deepcopy(LABEL_INPUTS),
    }


def _sampler(select: str) -> dict[str, Any]:
    init_args: dict[str, Any] = {
        "frequent_key": "freq",
        "num_frequent": TIMING.num_frequent,
        "infrequent_key": "infreq",
        "num_infrequent": NUM_INFREQUENT,
        "output_key": "sentinel2_l2a",
        "select": select,
        # Use a stand-in weekly mosaic for the latest quarterly mosaic when the pick
        # starts at least half a quarter after the first weekly mosaic (only possible in
        # training windows).
        "stand_in_min_gap": STAND_IN_MIN_GAP,
        # Every pixel in a window has the same change date, so this only matters if the
        # change is after the time series, which the windows rule out; it is set to be
        # safe.
        "category_target": "category",
        "change_day_target": "timestep",
    }
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
            "target/timestep/days",
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
        # Each change day maps to the earliest timestep whose midpoint is on or after it.
        "init_args": {"input_key": "sentinel2_l2a", "mode": "AFTER"},
    },
]


def _task(categories: list[str]) -> dict[str, Any]:
    num_classes = len(categories) + 1
    return {
        "class_path": "rslearn.train.tasks.multi_task.MultiTask",
        "init_args": {
            "input_mapping": {
                "category": {"label_category": "targets"},
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


def _model(head: str, num_classes: int) -> dict[str, Any]:
    return {
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
                        "category": _category_decoder(head, num_classes),
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
    }


def _make_config(
    source: str,
    head: str,
    project_name: str,
    run_name: str,
    ds_path: str | None = None,
    seed: int = 0,
) -> dict:
    categories = SOURCE_DATASETS[source].categories
    eval_split = {
        "transforms": [_sampler("latest"), NORMALIZE],
        "tags": {"split": "val"},
    }
    return {
        "seed_everything": seed,
        "model": _model(head, len(categories) + 1),
        "data": {
            "class_path": "rslearn.train.data_module.RslearnDataModule",
            "init_args": {
                "path": ds_path or dataset_path(source),
                "inputs": _inputs("frequent", "infrequent"),
                "task": _task(categories),
                "batch_size": 8,
                "num_workers": 16,
                "train_config": {
                    "transforms": [_sampler("random"), FLIP, NORMALIZE],
                    "tags": {"split": "train"},
                },
                "val_config": eval_split,
                "test_config": copy.deepcopy(eval_split),
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
        "project_name": project_name,
        "run_name": run_name,
        "management_dir": "${RSLP_PREFIX}/projects",
    }


def make_train_config(source: str, run: Run, ds_path: str | None = None) -> dict:
    """Make the training config for one source dataset and run."""
    return _make_config(
        source,
        run.head,
        PROJECT_NAME,
        f"{source}_{run.name}",
        ds_path=ds_path,
        seed=run.seed,
    )


def make_old_run_config(source: str, project_name: str, run_name: str) -> dict:
    """Make the config of a round 1 or 2 history run, on the round 3 dataset.

    Rounds 1 and 2 used the same model as round 3, so only the category head differs
    between runs. Only the history runs can be tested on the round 3 scenarios, since
    the recent runs need 20 weekly mosaics.

    Args:
        source: the source dataset.
        project_name: 2026_10_07_change_alerts (round 1) or 2026_10_12_change_alerts
            (round 2).
        run_name: the run name in that project, e.g. "{source}_fixed4_history_pool".
    """
    if "_history_" not in run_name:
        raise ValueError(f"{run_name} is not a history run")
    head = run_name.rsplit("_", 1)[1]
    return _make_config(source, head, project_name, run_name)


def make_test_config(train_config: dict, end_days: int) -> dict:
    """Make the config to test a run on one round 3 test scenario.

    The project_name and run_name are kept so the best checkpoint of the run is
    loaded. The test split has one prediction window per scenario, with the end_days
    option set to the scenario.
    """
    config = copy.deepcopy(train_config)
    init_args = config["data"]["init_args"]
    init_args["test_config"] = {
        "transforms": [_sampler("latest"), NORMALIZE],
        "tags": {"split": "test", "end_days": end_days},
    }
    return config


def make_round2_test_config(train_config: dict, source: str, end_days: int) -> dict:
    """Make the config to test a run on one round 2 test scenario.

    The round 2 test_history dataset has the frequent layer freq_test_{E} (the latest 4
    weekly mosaics within 63 days, ending E days after the change) and the quarterly
    layer infreq_test. With select: latest, the sampler uses all of freq_test_{E} and
    the latest 8 quarterly mosaics that end before it, as the round 2 sampler did. The
    two layers overlap, so DropOverlappingInfrequent first drops the quarterly mosaics
    that do not end strictly before the first weekly mosaic (see round2_filter.py).
    """
    config = copy.deepcopy(train_config)
    init_args = config["data"]["init_args"]
    init_args["path"] = f"{ROUND2_DATASET_ROOT}/{source}/test_history"
    init_args["inputs"] = _inputs(f"freq_test_{end_days}", "infreq_test")
    drop_overlapping = {
        "class_path": "round2_filter.DropOverlappingInfrequent",
        "init_args": {"frequent_key": "freq", "infrequent_key": "infreq"},
    }
    init_args["test_config"] = {
        "transforms": [drop_overlapping, _sampler("latest"), NORMALIZE],
        "tags": {"split": "test"},
    }
    # Only the test split is used, but the other splits must not reference the round 3
    # layers.
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
