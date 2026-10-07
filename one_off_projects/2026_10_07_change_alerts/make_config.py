"""Write the dataset config.json for the training or test datasets.

Training dataset (--mode train):
- freq_fixed_{k} / infreq_fixed_{k} for k in 0..3: slots ending 7, 35, 62, and 90
  days after the change.
- freq_rand_{k} / infreq_rand_{k} for k in 0..3: slots ending at a per-window random
  offset. In config.json they end at the change date (offset 0); they must be
  prepared with prepare_randomized.py, which shifts each window's time range by its
  random offset. Do not prepare them with `rslearn dataset prepare`.

Test datasets (--mode test_history or test_recent):
- freq_test_{E} (and infreq_test_{E} for test_history) for E in 7, 45, 90: time
  series ending E days after the change.

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
    HISTORY_LOOKBACK,
    IMAGE_BAND_SETS,
    INFREQUENT_DURATION,
    INFREQUENT_PERIOD,
    NUM_SLOTS,
    TEST_END_DAYS,
    TEST_HISTORY_FREQUENT_DURATION,
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


def _slot_layers(
    end_offsets: list[timedelta],
    frequent_duration: timedelta,
    with_infrequent: bool,
    slot_name_format: str,
) -> dict[str, dict[str, Any]]:
    infrequent_kwargs: dict[str, Any] = {}
    if with_infrequent:
        infrequent_kwargs = dict(
            infrequent_period=INFREQUENT_PERIOD,
            infrequent_duration=INFREQUENT_DURATION,
            infrequent_end_before=HISTORY_LOOKBACK,
        )
    return make_slot_layer_configs(
        end_offsets=end_offsets,
        data_source=DATA_SOURCE,
        band_sets=IMAGE_BAND_SETS,
        frequent_period=FREQUENT_PERIOD,
        frequent_duration=frequent_duration,
        frequent_layer_name="freq_" + slot_name_format,
        infrequent_layer_name="infreq_" + slot_name_format,
        **infrequent_kwargs,
    )


def make_config(mode: str) -> dict[str, Any]:
    """Make the dataset config for the given mode."""
    layers: dict[str, dict[str, Any]] = dict(LABEL_LAYERS)
    if mode == "train":
        fixed_offsets = slot_end_offsets(NUM_SLOTS, DETECTION_RANGE, FREQUENT_PERIOD)
        layers.update(
            _slot_layers(fixed_offsets, FREQUENT_DURATION, True, "fixed_{slot}")
        )
        layers.update(
            _slot_layers(
                [timedelta(0)] * NUM_SLOTS, FREQUENT_DURATION, True, "rand_{slot}"
            )
        )
    elif mode in ["test_history", "test_recent"]:
        test_offsets = [timedelta(days=days) for days in TEST_END_DAYS]
        frequent_duration = (
            TEST_HISTORY_FREQUENT_DURATION
            if mode == "test_history"
            else FREQUENT_DURATION
        )
        layers.update(
            make_slot_layer_configs(
                end_offsets=test_offsets,
                data_source=DATA_SOURCE,
                band_sets=IMAGE_BAND_SETS,
                frequent_period=FREQUENT_PERIOD,
                frequent_duration=frequent_duration,
                infrequent_period=INFREQUENT_PERIOD if mode == "test_history" else None,
                infrequent_duration=(
                    INFREQUENT_DURATION if mode == "test_history" else None
                ),
                infrequent_end_before=HISTORY_LOOKBACK,
                frequent_layer_name="freq_test_{slot}",
                infrequent_layer_name="infreq_test_{slot}",
                slot_names=[str(days) for days in TEST_END_DAYS],
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
