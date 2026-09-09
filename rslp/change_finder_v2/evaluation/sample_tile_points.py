"""Sample v2 annotation candidates from evaluation tiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import shapely
import tqdm
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

DEFAULT_POINTS_PER_TILE = 50
DEFAULT_GROUP = "evaluation_tiles"
DEFAULT_WINDOW_SIZE = 128
DEFAULT_RESOLUTION = 10


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


def _load_tiles(tiles_csv: Path) -> list[Tile]:
    """Load tile rows from tiles.csv."""
    with tiles_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        tiles: list[Tile] = []
        for row in reader:
            tile_id = row["tile_id"].strip()

            tiles.append(
                Tile(
                    tile_id=tile_id,
                    west=float(row["west"]),
                    south=float(row["south"]),
                    east=float(row["east"]),
                    north=float(row["north"]),
                    year=int(row["year"]),
                    compare_from_year=int(row["compare_from_year"]),
                    compare_to_year=int(row["compare_to_year"]),
                    raw={k: (v or "") for k, v in row.items()},
                )
            )

    return tiles


def _window_name(tile: Tile, point_idx: int, lon: float, lat: float) -> str:
    """Create a stable unique window name for a sampled point."""
    digest = hashlib.sha256(
        f"{tile.tile_id}:{point_idx}:{lon:.8f}:{lat:.8f}".encode()
    ).hexdigest()[:10]
    return f"{tile.tile_id}_{point_idx:03d}_{digest}"


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
    tile: Tile,
    point_idx: int,
    lon: float,
    lat: float,
    group: str,
    window_size: int,
    resolution: float,
) -> dict[str, Any]:
    """Create one v2 annotation entry centered on a sampled lon/lat point."""
    projection = get_utm_ups_projection(lon, lat, resolution, -resolution)
    center = STGeometry(
        WGS84_PROJECTION, shapely.Point(lon, lat), time_range=None
    ).to_projection(projection)
    center_col = math.floor(center.shp.x)
    center_row = math.floor(center.shp.y)
    half = window_size // 2

    compare_end_exclusive = tile.compare_to_year + 1
    return {
        "projection": projection.serialize(),
        "bounds": [
            center_col - half,
            center_row - half,
            center_col + half,
            center_row + half,
        ],
        "window_name": _window_name(tile, point_idx, lon, lat),
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
    tiles_csv: Path,
    points_per_tile: int = DEFAULT_POINTS_PER_TILE,
    group: str = DEFAULT_GROUP,
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

    tiles = _load_tiles(tiles_csv)
    entries: list[dict[str, Any]] = []

    for tile in tqdm.tqdm(tiles, desc="Sampling points"):
        for point_idx in range(points_per_tile):
            lon = random.uniform(tile.west, tile.east)
            lat = random.uniform(tile.south, tile.north)
            entries.append(
                _make_entry(
                    tile=tile,
                    point_idx=point_idx,
                    lon=lon,
                    lat=lat,
                    group=group,
                    window_size=window_size,
                    resolution=resolution,
                )
            )

    random.shuffle(entries)
    for output_idx, entry in enumerate(entries):
        entry["metadata"]["output_index"] = output_idx
    return entries


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
        window_size=args.window_size,
        resolution=args.resolution,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(entries, f, indent=2)
    print(
        f"Wrote {len(entries)} shuffled v2 annotation entries to {args.output} "
        f"from {args.tiles_csv}"
    )


if __name__ == "__main__":
    main()
