"""Write the dataset config.json, shared by all windows (train, val, and test).

The image layers come from rslearn.change_alerts.layers.make_layer_configs:

- frequent: weekly mosaics covering the window time range (up to 16 for a training
  window, up to 4 for a prediction window).
- infrequent: the 8 quarterly mosaics before the window start.

The dataset also has the label_category and label_change_day label layers, and the
output_category and output_change_day layers, which RslearnWriter needs (also during
`model fit`) if the model config writes predictions.
"""

import argparse
import json
from typing import Any

from common import (
    DATA_SOURCE,
    IMAGE_BAND_SETS,
    INFREQUENT_DURATION,
    INFREQUENT_PERIOD,
    TIMING,
)
from rslearn.change_alerts.layers import make_layer_configs
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

OUTPUT_LAYERS = {
    "output_category": {
        "type": "raster",
        "band_sets": [{"bands": ["output"], "dtype": "uint8"}],
    },
    "output_change_day": {
        "type": "raster",
        "band_sets": [{"bands": ["output"], "dtype": "uint16", "nodata_value": 65535}],
    },
}


def make_config() -> dict[str, Any]:
    """Make the dataset config."""
    layers: dict[str, dict[str, Any]] = make_layer_configs(
        TIMING,
        data_source=DATA_SOURCE,
        band_sets=IMAGE_BAND_SETS,
        infrequent_period=INFREQUENT_PERIOD,
        infrequent_duration=INFREQUENT_DURATION,
    )
    layers.update(LABEL_LAYERS)
    layers.update(OUTPUT_LAYERS)
    return {"layers": layers}


def main() -> None:
    """Write the config."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    with out_path.open("w") as f:
        json.dump(make_config(), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
