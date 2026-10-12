"""Write the dataset config.json for the training or test datasets.

Each context has its own frequent layers: freq_hist_* hold the latest 4 weekly mosaics
within 63 days, and freq_recent_* the latest 20 weekly mosaics within 26 weeks.

Training dataset (--mode train):
- freq_{ctx}_fixed_{k} for k in 0..3: slots ending 7, 35, 62, and 90 days after the
  change.
- freq_{ctx}_rand_{k} for k in 0..3: slots ending at a per-window random offset. In
  config.json they end at the change date (offset 0); they must be prepared with
  prepare_randomized.py, which shifts each window's time range by its random offset.
  Do not prepare them with `rslearn dataset prepare`.
- infreq: quarterly mosaics shared by all history slots (fixed and randomized, since
  the random offsets are also within 7 to 90 days).

Test datasets (--mode test_history or test_recent):
- freq_test_{E} for E in 7, 45, 90: time series ending E days after the change, with
  the history or recent frequent layer settings.
- infreq_test (test_history only): quarterly mosaics shared by the test slots.

All datasets also have the label_category and label_change_day label layers.
"""

import argparse
import json
from datetime import timedelta
from typing import Any

from common import (
    DATA_SOURCE,
    DETECTION_RANGE,
    FREQUENT_DURATION,
    FREQUENT_PERIOD,
    IMAGE_BAND_SETS,
    INFREQUENT_DURATION,
    INFREQUENT_PERIOD,
    LAYER_CTX,
    NUM_FREQUENT,
    NUM_SLOTS,
    TEST_END_DAYS,
)
from rslearn.change_alerts.slots import make_slot_layer_configs, slot_end_offsets
from upath import UPath

LABEL_LAYERS = {
    "label_category": {
        "type": "raster",
        "band_sets": [{"bands": ["label"], "dtype": "uint8"}],
    },
    "label_change_day": {
        "type": "raster",
        "band_sets": [{"bands": ["label"], "dtype": "uint16"}],
    },
}


def _frequent_layers(
    ctx: str,
    end_offsets: list[timedelta],
    layer_name: str,
    slot_names: list[str] | None = None,
    infrequent_layer_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Make the frequent layers for one context, and optionally the infrequent layer.

    Args:
        ctx: "history" or "recent".
        end_offsets: the offset from the change date to the end of each slot.
        layer_name: format string for the frequent layer names, with a {slot}
            placeholder.
        slot_names: the names to substitute for {slot}, defaults to the slot index.
        infrequent_layer_name: if set, also make an infrequent layer with this name,
            shared by the slots.
    """
    infrequent_kwargs: dict[str, Any] = {}
    if infrequent_layer_name is not None:
        infrequent_kwargs = {
            "infrequent_period": INFREQUENT_PERIOD,
            "infrequent_duration": INFREQUENT_DURATION,
            "infrequent_layer_name": infrequent_layer_name,
        }
    return make_slot_layer_configs(
        end_offsets=end_offsets,
        data_source=DATA_SOURCE,
        band_sets=IMAGE_BAND_SETS,
        frequent_period=FREQUENT_PERIOD,
        frequent_duration=FREQUENT_DURATION[ctx],
        frequent_max_matches=NUM_FREQUENT[ctx],
        frequent_layer_name=layer_name,
        slot_names=slot_names,
        **infrequent_kwargs,
    )


def make_config(mode: str) -> dict[str, Any]:
    """Make the dataset config for the given mode."""
    layers: dict[str, dict[str, Any]] = dict(LABEL_LAYERS)
    if mode == "train":
        fixed_offsets = slot_end_offsets(NUM_SLOTS, DETECTION_RANGE, FREQUENT_PERIOD)
        rand_offsets = [timedelta(0)] * NUM_SLOTS
        for ctx, layer_ctx in LAYER_CTX.items():
            layers.update(
                _frequent_layers(
                    ctx,
                    fixed_offsets,
                    f"freq_{layer_ctx}_fixed_{{slot}}",
                    infrequent_layer_name="infreq" if ctx == "history" else None,
                )
            )
            layers.update(
                _frequent_layers(ctx, rand_offsets, f"freq_{layer_ctx}_rand_{{slot}}")
            )
    elif mode in ["test_history", "test_recent"]:
        ctx = mode.removeprefix("test_")
        layers.update(
            _frequent_layers(
                ctx,
                [timedelta(days=days) for days in TEST_END_DAYS],
                "freq_test_{slot}",
                slot_names=[str(days) for days in TEST_END_DAYS],
                infrequent_layer_name="infreq_test" if ctx == "history" else None,
            )
        )
    else:
        raise ValueError(f"unknown mode {mode}")
    return {"layers": layers}


def main() -> None:
    """Write the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["train", "test_history", "test_recent"], required=True
    )
    parser.add_argument(
        "--ds_path",
        required=True,
        help="Dataset directory to write config.json in (or a .json filename)",
    )
    args = parser.parse_args()

    out_path = UPath(args.ds_path)
    if out_path.suffix != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "config.json"
    config = make_config(args.mode)
    with out_path.open("w") as f:
        json.dump(config, f, indent=2)
    image_layers = [name for name in config["layers"] if name not in LABEL_LAYERS]
    print(f"wrote {out_path} with image layers: {','.join(image_layers)}")


if __name__ == "__main__":
    main()
