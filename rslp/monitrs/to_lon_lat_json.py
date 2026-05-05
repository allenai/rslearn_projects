"""Build a longitude/latitude JSON list from the MONITRS dataset for from_lon_lat_list.

MONITRS annotations use ``lat_lon`` as ``[lat, lon]`` per timestep, we save as ``[[lon, lat], ...]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _lon_lat_from_record(record: object) -> tuple[float, float] | None:
    """Return (lon, lat) from a MONITRS JSON object, or None if missing or invalid."""
    if not isinstance(record, dict):
        return None
    lat_lon = record.get("lat_lon")
    if not isinstance(lat_lon, list) or not lat_lon:
        return None
    first = lat_lon[0]
    if not isinstance(first, list | tuple) or len(first) < 2:
        return None
    try:
        lat = float(first[0])
        lon = float(first[1])
    except (TypeError, ValueError):
        return None
    return (lon, lat)


def collect_monitrs_lon_lats(
    monitrs_dir: Path,
    splits: tuple[str, ...],
) -> list[list[float]]:
    """Load MONITRS split JSON files and return unique ``[lon, lat]`` pairs.

    Args:
        monitrs_dir: Directory containing ``total_train.json`` / ``total_test.json``.
        splits: Which splits to include; each element is ``"train"`` or ``"test"``.

    Returns:
        Sorted list of ``[longitude, latitude]`` pairs (deduplicated).
    """
    split_to_fname = {
        "train": "total_train.json",
        "test": "total_test.json",
    }
    seen: dict[tuple[float, float], None] = {}
    for split in splits:
        fname = split_to_fname.get(split)
        if fname is None:
            raise ValueError(
                f"unknown split {split!r}; expected one of {sorted(split_to_fname)}"
            )
        path = monitrs_dir / fname
        if not path.is_file():
            raise FileNotFoundError(f"expected MONITRS file at {path}")
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} should contain a JSON array of records")
        for record in data:
            pair = _lon_lat_from_record(record)
            if pair is None:
                continue
            seen.setdefault(pair, None)
    ordered = sorted(seen, key=lambda t: (t[0], t[1]))
    return [[lon, lat] for lon, lat in ordered]


def main() -> None:
    """Write JSON list of [lon, lat] pairs from MONITRS dataset."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a JSON list of [lon, lat] pairs from MONITRS for "
            "olmoearth_pretrain.dataset_creation.create_windows.from_lon_lat_list"
        ),
    )
    parser.add_argument(
        "--monitrs_dir",
        type=Path,
        default=Path("/weka/dfive-default/piperw/data/MONITRS"),
        help="Directory with total_train.json and total_test.json",
    )
    parser.add_argument(
        "--out_fname",
        type=Path,
        required=True,
        help="Output JSON path (list of [lon, lat])",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
        help="Which MONITRS JSON splits to scan (default: both)",
    )
    args = parser.parse_args()

    lon_lats = collect_monitrs_lon_lats(args.monitrs_dir, tuple(args.splits))
    args.out_fname.parent.mkdir(parents=True, exist_ok=True)
    with args.out_fname.open("w") as f:
        json.dump(lon_lats, f)
    print(
        f"wrote {len(lon_lats)} unique [lon, lat] pairs from splits {list(args.splits)} "
        f"to {args.out_fname}"
    )


if __name__ == "__main__":
    main()
