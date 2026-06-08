"""Sample v2 annotation candidates from evaluation tiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyproj.aoi
import pyproj.database
from pyproj import Transformer

DEFAULT_POINTS_PER_TILE = 50
DEFAULT_GROUP = "evaluation_tiles"
DEFAULT_SEED = 0
DEFAULT_WINDOW_SIZE = 128
DEFAULT_RESOLUTION = 10
UPS_NORTH_EPSG = 5041
UPS_SOUTH_EPSG = 5042
UPS_NORTH_THRESHOLD = 84 - 1e-4
UPS_SOUTH_THRESHOLD = -80 + 1e-4

REQUIRED_COLUMNS = ("tile_id", "west", "south", "east", "north", "year")


@dataclass(frozen=True)
class Tile:
    """One WGS84 evaluation tile from tiles.csv."""

    tile_id: str
    west: float
    south: float
    east: float
    north: float
    year: int
    compare_from_year: int
    compare_to_year: int
    raw: dict[str, str]


def _parse_float(row: dict[str, str], key: str, row_num: int) -> float:
    """Parse a required float CSV value."""
    value = row.get(key, "").strip()
    if value == "":
        raise ValueError(f"row {row_num} missing required column {key!r}")
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"row {row_num} has invalid {key!r}: {value!r}") from e


def _parse_int(row: dict[str, str], key: str, row_num: int) -> int:
    """Parse a required integer CSV value."""
    value = row.get(key, "").strip()
    if value == "":
        raise ValueError(f"row {row_num} missing required column {key!r}")
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"row {row_num} has invalid {key!r}: {value!r}") from e


def _parse_optional_int(
    row: dict[str, str], key: str, default: int, row_num: int
) -> int:
    """Parse an optional integer CSV value."""
    value = row.get(key, "").strip()
    if value == "":
        return default
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"row {row_num} has invalid {key!r}: {value!r}") from e


def _load_tiles(tiles_csv: Path) -> list[Tile]:
    """Load and validate tile rows."""
    with tiles_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{tiles_csv} missing required columns: {missing}")

        tiles: list[Tile] = []
        for row_num, row in enumerate(reader, start=2):
            tile_id = row["tile_id"].strip()
            if not tile_id:
                raise ValueError(f"row {row_num} missing tile_id")

            west = _parse_float(row, "west", row_num)
            south = _parse_float(row, "south", row_num)
            east = _parse_float(row, "east", row_num)
            north = _parse_float(row, "north", row_num)
            if east <= west or north <= south:
                raise ValueError(
                    f"row {row_num} has invalid bounds: "
                    f"{west},{south},{east},{north}"
                )

            year = _parse_int(row, "year", row_num)
            compare_from_year = _parse_optional_int(
                row, "compare_from_year", year - 1, row_num
            )
            compare_to_year = _parse_optional_int(
                row, "compare_to_year", year + 1, row_num
            )
            if compare_to_year < compare_from_year:
                raise ValueError(
                    f"row {row_num} has compare_to_year before compare_from_year"
                )

            tiles.append(
                Tile(
                    tile_id=tile_id,
                    west=west,
                    south=south,
                    east=east,
                    north=north,
                    year=year,
                    compare_from_year=compare_from_year,
                    compare_to_year=compare_to_year,
                    raw={k: (v or "") for k, v in row.items()},
                )
            )

    return tiles


def _safe_name(value: str) -> str:
    """Return a filesystem-friendly name component."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "tile"


def _window_name(tile: Tile, point_idx: int, lon: float, lat: float, seed: int) -> str:
    """Create a stable unique window name for a sampled point."""
    digest = hashlib.sha256(
        f"{tile.tile_id}:{point_idx}:{lon:.8f}:{lat:.8f}:{seed}".encode()
    ).hexdigest()[:10]
    return f"{_safe_name(tile.tile_id)}_{point_idx:03d}_{digest}"


def _get_utm_ups_epsg(lon: float, lat: float) -> int:
    """Get the UTM/UPS EPSG code matching rslearn's projection helper."""
    if lat > UPS_NORTH_THRESHOLD:
        return UPS_NORTH_EPSG
    if lat < UPS_SOUTH_THRESHOLD:
        return UPS_SOUTH_EPSG

    infos = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    if not infos:
        raise ValueError(f"could not find UTM zone for lon={lon}, lat={lat}")
    return int(infos[0].code)


def _project_lonlat(
    lon: float, lat: float, epsg: int, resolution: float
) -> tuple[int, int]:
    """Project lon/lat to rslearn pixel coordinates."""
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x_meters, y_meters = transformer.transform(lon, lat)
    return math.floor(x_meters / resolution), math.floor(y_meters / -resolution)


def _tile_metadata(tile: Tile, point_idx: int) -> dict[str, Any]:
    """Create compact per-entry metadata copied from the tile CSV."""
    metadata: dict[str, Any] = {
        "source": "change_finder_v2/evaluation/tiles.csv",
        "tile_id": tile.tile_id,
        "tile_bounds": [tile.west, tile.south, tile.east, tile.north],
        "tile_year": tile.year,
        "compare_from_year": tile.compare_from_year,
        "compare_to_year": tile.compare_to_year,
        "point_index_within_tile": point_idx,
    }

    for key in (
        "macro_region",
        "country",
        "site",
        "change_type",
        "expected_change",
        "evidence_urls",
        "notes",
    ):
        value = tile.raw.get(key, "").strip()
        if value:
            metadata[key] = value

    return metadata


def _make_entry(
    *,
    tile: Tile,
    point_idx: int,
    lon: float,
    lat: float,
    group: str,
    seed: int,
    window_size: int,
    resolution: float,
) -> dict[str, Any]:
    """Create one v2 annotation entry centered on a sampled lon/lat point."""
    epsg = _get_utm_ups_epsg(lon, lat)
    center_col, center_row = _project_lonlat(lon, lat, epsg, resolution)
    half = window_size // 2

    compare_end_exclusive = tile.compare_to_year + 1
    return {
        "projection": {
            "crs": f"EPSG:{epsg}",
            "x_resolution": resolution,
            "y_resolution": -resolution,
        },
        "bounds": [
            center_col - half,
            center_row - half,
            center_col + half,
            center_row + half,
        ],
        "window_name": _window_name(tile, point_idx, lon, lat, seed),
        "group": group,
        "time_range": [
            f"{tile.compare_from_year:04d}-01-01T00:00:00+00:00",
            f"{compare_end_exclusive:04d}-01-01T00:00:00+00:00",
        ],
        "positive_points": [{"lon": lon, "lat": lat}],
        "negative_points": [],
        "metadata": _tile_metadata(tile, point_idx),
    }


def sample_entries(
    *,
    tiles_csv: Path,
    points_per_tile: int = DEFAULT_POINTS_PER_TILE,
    group: str = DEFAULT_GROUP,
    seed: int = DEFAULT_SEED,
    window_size: int = DEFAULT_WINDOW_SIZE,
    resolution: float = DEFAULT_RESOLUTION,
) -> list[dict[str, Any]]:
    """Sample candidate points and return shuffled v2 annotation entries."""
    if points_per_tile < 0:
        raise ValueError("points_per_tile must be non-negative")
    if window_size <= 0 or window_size % 2 != 0:
        raise ValueError("window_size must be a positive even integer")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    rng = random.Random(seed)
    tiles = _load_tiles(tiles_csv)
    entries: list[dict[str, Any]] = []

    for tile in tiles:
        lon_span = tile.east - tile.west
        lat_span = tile.north - tile.south
        for point_idx in range(points_per_tile):
            lon = tile.west + rng.random() * lon_span
            lat = tile.south + rng.random() * lat_span
            entries.append(
                _make_entry(
                    tile=tile,
                    point_idx=point_idx,
                    lon=lon,
                    lat=lat,
                    group=group,
                    seed=seed,
                    window_size=window_size,
                    resolution=resolution,
                )
            )

    rng.shuffle(entries)
    for output_idx, entry in enumerate(entries):
        entry["metadata"]["output_index"] = output_idx
    return entries


def write_entries(entries: list[dict[str, Any]], output: Path) -> None:
    """Write v2 annotation entries as pretty-printed JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")


def main() -> None:
    """CLI entrypoint."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Sample random points from evaluation/tiles.csv and write a shuffled "
            "change_finder_v2 annotation JSON."
        )
    )
    parser.add_argument(
        "--tiles-csv",
        type=Path,
        default=script_dir / "tiles.csv",
        help="Input tile CSV. Default: tiles.csv next to this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "tile_points_v2.json",
        help="Output v2 annotation JSON. Default: tile_points_v2.json next to this script.",
    )
    parser.add_argument(
        "-n",
        "--points-per-tile",
        type=int,
        default=DEFAULT_POINTS_PER_TILE,
        help=f"Number of random points to sample per tile. Default: {DEFAULT_POINTS_PER_TILE}.",
    )
    parser.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help=f"Window group to assign in the v2 JSON. Default: {DEFAULT_GROUP}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for sampling and shuffling. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Annotation window size in pixels. Default: {DEFAULT_WINDOW_SIZE}.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help=f"UTM/UPS resolution in meters per pixel. Default: {DEFAULT_RESOLUTION:g}.",
    )
    args = parser.parse_args()

    entries = sample_entries(
        tiles_csv=args.tiles_csv,
        points_per_tile=args.points_per_tile,
        group=args.group,
        seed=args.seed,
        window_size=args.window_size,
        resolution=args.resolution,
    )
    write_entries(entries, args.output)
    print(
        f"Wrote {len(entries)} shuffled v2 annotation entries to {args.output} "
        f"from {args.tiles_csv}"
    )


if __name__ == "__main__":
    main()
