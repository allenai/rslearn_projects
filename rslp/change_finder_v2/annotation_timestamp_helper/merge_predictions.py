"""Merge timestamp helper predictions back into copied annotation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rslearn.dataset import Dataset
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from .constants import DATE_FIELDS, MANIFEST_FNAME, OUTPUT_LAYER, TIMESTAMP_HEADS


def _load_prediction_props(dataset: Dataset, group: str, name: str) -> dict[str, Any]:
    """Load the prediction feature properties for one window."""
    windows = dataset.load_windows(groups=[group], names=[name])
    if len(windows) != 1:
        raise ValueError(f"expected one window for {group}/{name}, got {len(windows)}")
    window = windows[0]
    features = GeojsonVectorFormat().decode_vector(
        window.get_layer_dir(OUTPUT_LAYER), window.projection, window.bounds
    )
    if not features:
        raise ValueError(f"no prediction features found for {group}/{name}")
    return features[0].properties


def merge_predictions(
    *,
    dataset_path: str,
    json_paths: list[str],
    output_dir: str | None = None,
    output_json: str | None = None,
    allow_missing: bool = False,
) -> None:
    """Merge predictions into copied JSON annotations."""
    if output_dir is None and output_json is None:
        raise ValueError("set either --output-dir or --output-json")
    if output_json is not None and len(json_paths) != 1:
        raise ValueError("--output-json can only be used with exactly one --json")

    selected = {str(Path(path).resolve()): path for path in json_paths}
    outputs: dict[str, list[dict[str, Any]]] = {}
    for resolved, path in selected.items():
        with open(path) as f:
            outputs[resolved] = json.load(f)

    with (UPath(dataset_path) / MANIFEST_FNAME).open() as f:
        manifest = json.load(f)

    dataset = Dataset(UPath(dataset_path))
    updated = 0
    for record in manifest["windows"]:
        source_json = record["source_json"]
        if record["source_role"] != "inference" or source_json not in outputs:
            continue

        try:
            props = _load_prediction_props(
                dataset, record["group"], record["window_name"]
            )
        except Exception:
            if allow_missing:
                continue
            raise

        entry = outputs[source_json][record["entry_index"]]
        point = entry["positive_points"][record["point_index"]]
        for head in TIMESTAMP_HEADS:
            date_key = f"{head}_date"
            if date_key not in props:
                raise ValueError(
                    f"prediction for {record['group']}/{record['window_name']} "
                    f"is missing {date_key}"
                )
            point[DATE_FIELDS[head]] = props[date_key]
        updated += 1

    if output_json is not None:
        out_path = Path(output_json)
        source = next(iter(outputs))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(outputs[source], f, indent=2)
        print(f"Wrote {out_path}")
    else:
        assert output_dir is not None
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for resolved, entries in outputs.items():
            out_path = out_dir / Path(selected[resolved]).name
            with out_path.open("w") as f:
                json.dump(entries, f, indent=2)
            print(f"Wrote {out_path}")
    print(f"Updated {updated} inference points")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Merge annotation timestamp helper predictions into JSON copies."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--json", action="append", required=True, dest="json_paths")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    merge_predictions(
        dataset_path=args.dataset_path,
        json_paths=args.json_paths,
        output_dir=args.output_dir,
        output_json=args.output_json,
        allow_missing=args.allow_missing,
    )


if __name__ == "__main__":
    main()
