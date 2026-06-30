"""Render per-row example figures from change_finder_v2 (LCC model) deployment outputs.

Each output GeoJSON (e.g. under
``/weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_outputs_20260617/``)
contains detected land-cover-change polygons with properties:

- ``src_class`` / ``dst_class``: source and destination land cover categories.
- ``timestamp_start`` / ``timestamp_end``: the estimated change time range.
- ``num_pixels`` / ``avg_change_score``: detection size and confidence.

For each selected polygon we render one figure row with five columns:

1. A metadata text box: Location, Change type (Src -> Dst), Start time, End time.
2. Sentinel-2 RGB imagery ~4 months *before* the change start.
3. Sentinel-2 RGB imagery at (roughly) the change start.
4. Sentinel-2 RGB imagery at (roughly) the change end (start + 1 month if start == end).
5. Sentinel-2 RGB imagery ~4 months *after* the change end.

Imagery is fetched live as 8-bit Sentinel-2 visual (R, G, B) mosaics via the
``Sentinel2L2A`` olmoearth_datasets data source. We build a temporary rslearn dataset
with one window per selected polygon spanning the four target dates, materialize monthly
least-cloudy mosaics, and pick the mosaic nearest each target date.

Each row is saved as its own PNG so the user can pick a handful for the paper.

Example::

    python -m rslp.change_finder_v2.deployment_figure.run \
        --outputs-dir /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_outputs_20260617 \
        --output-dir /weka/.../deployment_figure_rows \
        --limit 30 --min-pixels 150
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import tempfile
from datetime import UTC, date, datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
import shapely  # noqa: E402
import shapely.geometry  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rslearn.const import WGS84_PROJECTION  # noqa: E402
from rslearn.dataset import Dataset  # noqa: E402
from rslearn.dataset.add_windows import add_windows_from_geometries  # noqa: E402
from rslearn.utils import Projection, STGeometry  # noqa: E402
from upath import UPath  # noqa: E402

from rslp.utils.rslearn import (  # noqa: E402
    ApplyWindowsArgs,
    IngestArgs,
    MaterializeArgs,
    MaterializePipelineArgs,
    PrepareArgs,
    materialize_dataset,
)

# Months offset for the "before" and "after" context images.
CONTEXT_MONTHS = 4
# Buffer (days) added around the target span when defining each window's time range.
SPAN_BUFFER_DAYS = 50
# Sentinel-2 ground resolution (meters/pixel) for the rendered windows.
S2_RESOLUTION = 10.0
# Window size bounds (pixels) for the rendered crops.
MIN_WINDOW_PX = 48
MAX_WINDOW_PX = 192

LAYER = "sentinel2"
GROUP = "default"

CONFIG = {
    "layers": {
        LAYER: {
            "type": "raster",
            "band_sets": [{"bands": ["R", "G", "B"], "dtype": "uint8"}],
            "data_source": {
                "class_path": (
                    "olmoearth_run.runner.tools.rslearn_data_sources."
                    "olmoearth_datasets.sentinel2_l2a.Sentinel2L2A"
                ),
                "ingest": False,
                "init_args": {
                    "harmonize": False,
                    "query": {"sort_by": "CLOUD_COVER", "sort_direction": "ASC"},
                },
                "query_config": {
                    "space_mode": "MOSAIC",
                    "period_duration": "30d",
                    "max_matches": 120,
                    "per_period_mosaic_reverse_time_order": False,
                },
            },
        }
    }
}


def parse_dt(s: str) -> datetime:
    """Parse an ISO timestamp into a UTC datetime."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def add_months(dt: datetime, months: int) -> datetime:
    """Return ``dt`` shifted by ``months`` (positive or negative) months."""
    month_index = (dt.year * 12 + (dt.month - 1)) + months
    year, month = divmod(month_index, 12)
    month += 1
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days_in_month = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(dt.day, days_in_month[month - 1])
    return dt.replace(year=year, month=month, day=day)


def cap_cat(s: str) -> str:
    """Capitalize each space-separated word, preserving hyphens and slashes."""
    return "/".join(
        " ".join(w[:1].upper() + w[1:] for w in part.split(" "))
        for part in s.split("/")
    )


def fdate(d: datetime | date) -> str:
    """Format a date as e.g. "Mar 2023"."""
    return d.strftime("%b %Y")


def utm_epsg(lon: float, lat: float) -> int:
    """Return the UTM EPSG code for a lon/lat."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return (32600 if lat >= 0 else 32700) + zone


_GEOCODE_CACHE: dict[tuple[float, float], str] = {}


def reverse_geocode(lon: float, lat: float) -> str:
    """Return a human-readable location name, falling back to lat/lon."""
    key = (round(lat, 3), round(lon, 3))
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]
    name = f"{lat:.3f}, {lon:.3f}"
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 10},
            headers={"User-Agent": "rslp-change-finder-figure/1.0"},
            timeout=20,
        )
        if resp.ok:
            addr = resp.json().get("address", {})
            parts = [
                addr.get("city") or addr.get("town") or addr.get("village")
                or addr.get("county") or addr.get("state"),
                addr.get("country"),
            ]
            label = ", ".join(p for p in parts if p)
            if label:
                name = label
    except Exception:  # noqa: BLE001 - geocoding is best-effort
        pass
    _GEOCODE_CACHE[key] = name
    return name


def target_dates(start: datetime, end: datetime) -> list[tuple[str, datetime]]:
    """Return the four (label, target date) image columns for a change."""
    img_end = end if end > start else add_months(start, 1)
    return [
        ("4 months before", add_months(start, -CONTEXT_MONTHS)),
        ("Change start", start),
        ("Change end", img_end),
        ("4 months after", add_months(img_end, CONTEXT_MONTHS)),
    ]


def collect_candidates(
    outputs_dir: str,
    min_pixels: int,
    max_per_type: int,
    limit: int,
    max_abs_lat: float | None = None,
) -> list[dict]:
    """Collect candidate change polygons across all output GeoJSONs."""
    candidates: list[dict] = []
    for path in sorted(glob.glob(os.path.join(outputs_dir, "*.geojson"))):
        try:
            data = json.load(open(path))
        except Exception:  # noqa: BLE001
            continue
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            num_pixels = props.get("num_pixels", 0)
            if num_pixels < min_pixels:
                continue
            src = props.get("src_class")
            dst = props.get("dst_class")
            if not src or not dst or src == dst:
                continue
            try:
                geom = shapely.geometry.shape(feat["geometry"])
            except Exception:  # noqa: BLE001
                continue
            centroid = geom.centroid
            minx, miny, maxx, maxy = geom.bounds
            lat = float(centroid.y)
            if max_abs_lat is not None and abs(lat) > max_abs_lat:
                continue
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
            width_m = (maxx - minx) * m_per_deg_lon
            height_m = (maxy - miny) * m_per_deg_lat
            half_size_m = max(width_m, height_m) / 2.0 * 1.8
            half_size_m = max(60.0, min(900.0, half_size_m))
            candidates.append(
                {
                    "lon": float(centroid.x),
                    "lat": lat,
                    "half_size_m": half_size_m,
                    "src": src,
                    "dst": dst,
                    "start": props.get("timestamp_start"),
                    "end": props.get("timestamp_end"),
                    "num_pixels": num_pixels,
                    "score": props.get("avg_change_score", 0),
                }
            )

    candidates.sort(key=lambda c: c["num_pixels"], reverse=True)

    per_type: dict[tuple[str, str], int] = {}
    selected: list[dict] = []
    for cand in candidates:
        key = (cand["src"], cand["dst"])
        if per_type.get(key, 0) >= max_per_type:
            continue
        per_type[key] = per_type.get(key, 0) + 1
        selected.append(cand)
        if len(selected) >= limit:
            break
    return selected


def setup_windows(dataset: Dataset, candidates: list[dict]) -> dict[int, str]:
    """Create one window per candidate; return {candidate index: window name}."""
    window_names: dict[int, str] = {}
    for i, cand in enumerate(candidates):
        lon, lat = cand["lon"], cand["lat"]
        start = parse_dt(cand["start"])
        end = parse_dt(cand["end"])
        targets = target_dates(start, end)
        span_start = targets[0][1] - timedelta(days=SPAN_BUFFER_DAYS)
        span_end = targets[-1][1] + timedelta(days=SPAN_BUFFER_DAYS)

        window_px = int(round(2 * cand["half_size_m"] / S2_RESOLUTION))
        window_px = max(MIN_WINDOW_PX, min(MAX_WINDOW_PX, window_px))

        crs = CRS.from_epsg(utm_epsg(lon, lat))
        projection = Projection(crs, S2_RESOLUTION, -S2_RESOLUTION)
        name = f"cand_{i:04d}"
        geometry = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
        add_windows_from_geometries(
            dataset=dataset,
            group=GROUP,
            geometries=[geometry],
            projection=projection,
            name=name,
            window_size=window_px,
            time_range=(span_start, span_end),
        )
        window_names[i] = name
    return window_names


def read_materialized_images(
    dataset: Dataset, name: str
) -> list[tuple[datetime, np.ndarray]]:
    """Return (mid-time, HWC uint8 RGB) for each materialized mosaic in a window."""
    windows = dataset.load_windows(groups=[GROUP], names=[name])
    if not windows:
        return []
    window = windows[0]
    layer_datas = window.load_layer_datas()
    layer_cfg = dataset.layers[LAYER]
    band_cfg = layer_cfg.band_sets[0]
    projection, bounds = band_cfg.get_final_projection_and_bounds(
        window.projection, window.bounds
    )
    raster_format = band_cfg.instantiate_raster_format()

    images: list[tuple[datetime, np.ndarray]] = []
    for layer_name, group_idx in window.list_completed_layers():
        if layer_name != LAYER:
            continue
        raster_dir = window.get_raster_dir(layer_name, band_cfg.bands, group_idx)
        raster = raster_format.decode_raster(raster_dir, projection, bounds)
        chw = raster.get_chw_array()
        hwc = np.transpose(chw[:3], (1, 2, 0)).astype(np.uint8)

        # Recover the mosaic mid-time from the item provenance.
        mid = None
        layer_data = layer_datas.get(layer_name)
        if layer_data is not None and group_idx < len(layer_data.serialized_item_groups):
            item_group = layer_data.serialized_item_groups[group_idx]
            if item_group:
                tr = item_group[0].get("geometry", {}).get("time_range")
                if tr is not None:
                    t0 = datetime.fromisoformat(tr[0])
                    t1 = datetime.fromisoformat(tr[1])
                    mid = t0 + (t1 - t0) / 2
        if mid is None:
            continue
        images.append((mid, hwc))
    images.sort(key=lambda x: x[0])
    return images


def bad_fraction(img: np.ndarray) -> float:
    """Return the fraction of cloudy (very bright) or nodata (near-zero) pixels."""
    bright = np.all(img >= 238, axis=-1)
    dark = np.all(img <= 3, axis=-1)
    return float(np.mean(bright | dark))


def pick_nearest(
    images: list[tuple[datetime, np.ndarray]], target: datetime
) -> tuple[datetime, np.ndarray] | None:
    """Return the (time, image) best matching ``target`` in date and clarity.

    Balances temporal proximity to ``target`` against image clarity so we avoid
    picking a fully-clouded mosaic that merely happens to be closest in date: a
    clean mosaic ~45 days away is preferred over a fully-clouded one at the target.
    """
    if not images:
        return None

    def score(im: tuple[datetime, np.ndarray]) -> float:
        days = abs((im[0] - target).days)
        return days / 45.0 + 5.0 * bad_fraction(im[1])

    return min(images, key=score)


def render_row(
    cand: dict, panels: list[tuple[str, np.ndarray]], output_path: str
) -> None:
    """Render and save one figure row to ``output_path``."""
    location = cand["location"]
    start = parse_dt(cand["start"])
    end = parse_dt(cand["end"])
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), gridspec_kw={
        "width_ratios": [1.25, 1, 1, 1, 1], "wspace": 0.04
    })

    meta_ax = axes[0]
    meta_ax.axis("off")
    lines = [
        ("Location", location),
        ("Change", f"{cap_cat(cand['src'])} \u2192 {cap_cat(cand['dst'])}"),
        ("Start time", fdate(start)),
        ("End time", fdate(end)),
    ]
    y = 0.82
    for label, value in lines:
        meta_ax.text(0.04, y, label, fontsize=11, fontweight="bold",
                     va="top", ha="left", transform=meta_ax.transAxes)
        meta_ax.text(0.04, y - 0.08, value, fontsize=11, va="top", ha="left",
                     transform=meta_ax.transAxes, wrap=True)
        y -= 0.21

    for ax, (title, img) in zip(axes[1:], panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        default=(
            "/weka/dfive-default/rslearn-eai/datasets/change_finder/"
            "lcc_model_outputs_20260617"
        ),
        help="Directory of output GeoJSONs.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write row PNGs into."
    )
    parser.add_argument(
        "--dataset-dir", default=None,
        help="Where to build the temporary rslearn dataset (default: a temp dir).",
    )
    parser.add_argument("--limit", type=int, default=30,
                        help="Max number of rows to render.")
    parser.add_argument("--min-pixels", type=int, default=150,
                        help="Minimum num_pixels for a polygon to be considered.")
    parser.add_argument("--max-per-type", type=int, default=4,
                        help="Max rows per (src, dst) change type.")
    parser.add_argument("--max-abs-lat", type=float, default=None,
                        help="Drop polygons whose |latitude| exceeds this (avoids "
                             "snowy high-latitude scenes).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Workers for rslearn prepare/materialize.")
    parser.add_argument("--no-geocode", action="store_true",
                        help="Disable reverse geocoding (use lat/lon as location).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    candidates = collect_candidates(
        args.outputs_dir, args.min_pixels, args.max_per_type, args.limit,
        max_abs_lat=args.max_abs_lat,
    )
    print(f"Selected {len(candidates)} candidate polygons", flush=True)
    if not candidates:
        return

    ds_dir = args.dataset_dir or tempfile.mkdtemp(prefix="deployment_figure_ds_")
    ds_path = UPath(ds_dir)
    ds_path.mkdir(parents=True, exist_ok=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Building dataset at {ds_dir}", flush=True)

    dataset = Dataset(ds_path)
    window_names = setup_windows(dataset, candidates)

    apply_args = ApplyWindowsArgs(workers=args.workers, group=GROUP)
    materialize_dataset(
        ds_path,
        MaterializePipelineArgs(
            disabled_layers=[],
            prepare_args=PrepareArgs(apply_windows_args=apply_args),
            ingest_args=IngestArgs(apply_windows_args=apply_args),
            materialize_args=MaterializeArgs(
                ignore_errors=True, apply_windows_args=apply_args
            ),
        ),
    )

    manifest = []
    rendered = 0
    for i, cand in enumerate(candidates):
        name = window_names[i]
        images = read_materialized_images(dataset, name)
        if not images:
            print(f"[{i}] no imagery for {cand['lat']:.4f},{cand['lon']:.4f}",
                  flush=True)
            continue
        start = parse_dt(cand["start"])
        end = parse_dt(cand["end"])
        panels: list[tuple[str, np.ndarray]] = []
        ok = True
        for label, target in target_dates(start, end):
            pick = pick_nearest(images, target)
            if pick is None:
                ok = False
                break
            mid, img = pick
            panels.append((f"{label}\n{fdate(mid)}", img))
        if not ok:
            continue

        cand["location"] = (
            reverse_geocode(cand["lon"], cand["lat"])
            if not args.no_geocode else f"{cand['lat']:.3f}, {cand['lon']:.3f}"
        )
        out_path = os.path.join(args.output_dir, f"row_{i:03d}.png")
        render_row(cand, panels, out_path)
        manifest.append({
            "index": i,
            "file": os.path.basename(out_path),
            "location": cand["location"],
            "src": cand["src"],
            "dst": cand["dst"],
            "start": start.isoformat(),
            "end": end.isoformat(),
            "lon": cand["lon"],
            "lat": cand["lat"],
            "num_pixels": cand["num_pixels"],
        })
        rendered += 1
        print(f"[{i}] wrote {out_path} ({cand['src']} -> {cand['dst']}, "
              f"{cand['location']})", flush=True)

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Rendered {rendered} rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
