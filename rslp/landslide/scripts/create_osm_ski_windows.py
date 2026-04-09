r"""Create windows for false positives of landslide detection (segmentation task) - data source is OSM ski resorts.

Features are collected from OSM objects tagged with ski-related keys (``piste:type``,
``landuse=winter_sports``, ``aerialway``, etc.) using pyosmium's area-aware reader.

``*.osm.pbf`` files are huge - can extract a small region with: 
```osmium extract -b 6.0,45.5,10.5,47.5 \
  /path/to/europe-latest.osm.pbf \
  -o /path/to/alps-ski-trial.osm.pbf```

With ``--max_samples``, the default ``--sample_mode reservoir`` keeps an *unbiased* random
subset but must scan the **entire** PBF. ``--sample_mode prefix`` stops right after N
accepted sites, but on standard extracts **nodes are stored before ways**, and polygon
``Area`` objects are built from ways. So for ``--areas_only`` on continent-scale files you
still pay a **full sequential read of the node section** (often tens of GB / long wall time)
before the first resort polygon can appear. For quick tests, **clip the PBF** with
``osmium extract -b ...`` or use a **country / regional** extract (see example below).

Example (full run, takes long time):
    PYTHONPATH=/path/to/rslearn_projects python rslp/landslide/scripts/create_osm_ski_windows.py \\
        --pbf_path /weka/dfive/default/piperw/data/europe-latest.osm.pbf \\
        --ds_path data/landslide/osm_ski_windows/ \\
        --max_samples 5000 \\
        --time_start 2019-06-01 \\
        --time_end 2019-08-31

Example (actually fast: small bbox clip, then prefix sampling):
    osmium extract -b 6.0,45.5,10.5,47.5 europe-latest.osm.pbf -o alps.osm.pbf
    PYTHONPATH=/path/to/rslearn_projects python rslp/landslide/scripts/create_osm_ski_windows.py \\
        --pbf_path alps.osm.pbf \\
        --ds_path data/landslide/osm_ski_windows_test/ \\
        --max_samples 20 \\
        --sample_mode prefix \\
        --areas_only \\
        --progress_every 2000000 \\
        --num_workers 4

``--ds_path`` must be an existing rslearn dataset directory containing ``config.json``, consistent with the landslide dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast

import osmium
import shapely
import shapely.ops
import tqdm
from osmium.filter import KeyFilter
from osmium.geom import WKTFactory
from pyproj import Transformer
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from shapely import from_wkt
from upath import UPath

from rslp.utils.windows import calculate_bounds

WINDOW_RESOLUTION = 10  # meters per pixel (match landslide chips)
WINDOW_SIZE_PIXELS = 64
LABEL_LAYER = "label"
DEFAULT_GROUP = "osm_ski"


def _tags_indicate_ski(tags: Mapping[str, str]) -> bool:
    """Return True if OSM tags describe ski pistes, lifts, or winter-sports landuse."""
    t = dict(tags)
    if t.get("landuse") == "winter_sports":
        return True
    if "piste:type" in t:
        return True
    if "aerialway" in t:
        return True
    if t.get("sport") in ("ski", "yes"):
        return True
    if t.get("leisure") == "pitch" and t.get("sport") in ("ski", "yes"):
        return True
    if t.get("tourism") == "alpine_hut":
        return True
    return False


def _closed_way_polygon_from_wkt(wkt: str) -> shapely.Geometry | None:
    """Build a polygon from a closed line WKT if coordinates form a ring."""
    geom = from_wkt(wkt)
    if not isinstance(geom, shapely.LineString):
        return None
    if not geom.is_ring or len(geom.coords) < 4:
        return None
    try:
        return shapely.Polygon(geom.coords)
    except Exception:
        return None


def _geometry_from_osm_object(obj: Any, wkt_factory: WKTFactory) -> shapely.Geometry | None:
    """Turn a pyosmium Node, Way, or Area into a Shapely geometry in WGS84 (lon/lat)."""
    type_name = type(obj).__name__

    if type_name == "Node":
        return shapely.Point(obj.location.lon, obj.location.lat)

    if type_name == "Area":
        try:
            wkt = wkt_factory.create_multipolygon(obj)
        except RuntimeError:
            return None
        g = from_wkt(wkt)
        return g if not g.is_empty else None

    if type_name == "Way":
        w = obj
        if not _tags_indicate_ski(w.tags):
            return None
        try:
            wkt = wkt_factory.create_linestring(w)
        except RuntimeError:
            return None
        ls = from_wkt(wkt)
        if not isinstance(ls, shapely.LineString) or ls.is_empty:
            return None
        # Closed landuse / piste polygons are emitted as Areas as well; skip duplicate closed
        # ways that are not lift lines (aerialway may legitimately close).
        if w.is_closed() and "aerialway" not in dict(w.tags):
            poly = _closed_way_polygon_from_wkt(wkt)
            if poly is not None:
                return None
        return ls

    return None


def _representative_point(geom: shapely.Geometry) -> shapely.Point:
    if isinstance(geom, shapely.Point):
        return geom
    try:
        return geom.representative_point()
    except Exception:
        return geom.centroid


def _geometry_area_m2(geom: shapely.Geometry, utm_epsg: int) -> float:
    """Approximate area in m^2 for filtering; lines and points yield 0."""
    try:
        transformer = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True
        )

        def _tr(x: float, y: float, z: float | None = None) -> tuple[float, float]:
            return transformer.transform(x, y)

        g = shapely.ops.transform(_tr, geom)
        if g.geom_type in ("Polygon", "MultiPolygon"):
            return float(g.area)
    except Exception:
        pass
    return 0.0


@dataclass
class SkiSiteRecord:
    """One ski-related OSM feature to place a window on."""

    osm_kind: str  # "node" | "way" | "area"
    osm_id: int
    geometry: shapely.Geometry
    tags: dict[str, str]


def iter_ski_sites_from_pbf(
    pbf_path: str,
    bbox: tuple[float, float, float, float] | None = None,
    min_polygon_area_m2: float = 0.0,
    areas_only: bool = False,
    progress_every: int = 0,
) -> Iterator[SkiSiteRecord]:
    """Stream ski-related features from a PBF path.

    Args:
        pbf_path: Path to ``.osm.pbf`` or ``.osm``.
        bbox: Optional ``(min_lon, min_lat, max_lon, max_lat)`` filter on representative point.
        min_polygon_area_m2: Drop polygon sites smaller than this area (meters), after UTM warp.
        areas_only: If True, only emit polygonal ``Area`` features (e.g. resort footprints),
            skipping nodes and open ways such as lifts and pistes as lines.
        progress_every: If > 0, print a heartbeat every this many raw (key-filtered) objects
            so long node-section scans do not look hung.
    """
    # Restrict to objects that might carry ski tags to speed up I/O.
    key_filter = KeyFilter(
        "piste:type",
        "landuse",
        "aerialway",
        "sport",
        "leisure",
        "tourism",
    )
    proc = osmium.FileProcessor(pbf_path).with_areas(key_filter)
    wkt_factory = WKTFactory()

    seen = 0
    emitted = 0
    for obj in proc:
        seen += 1
        if progress_every > 0 and seen % progress_every == 0:
            print(
                f"  ... PBF scan: {seen:,} key-filtered objects read, "
                f"{emitted} ski site(s) emitted so far ...",
                flush=True,
            )

        if not _tags_indicate_ski(obj.tags):
            continue

        type_name = type(obj).__name__
        if areas_only and type_name != "Area":
            continue
        if type_name == "Way" and obj.is_closed() and "aerialway" not in dict(obj.tags):
            # Polygonal features appear again as Area; keep the Area only.
            continue

        geom = _geometry_from_osm_object(obj, wkt_factory)
        if geom is None or geom.is_empty:
            continue

        pt = _representative_point(geom)
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            if not (min_lon <= pt.x <= max_lon and min_lat <= pt.y <= max_lat):
                continue

        tags = {t.k: t.v for t in obj.tags}

        if min_polygon_area_m2 > 0 and type_name == "Area":
            lon, lat = pt.x, pt.y
            utm = get_utm_ups_crs(lon, lat)
            epsg = int(utm.split(":")[-1])
            if _geometry_area_m2(geom, epsg) < min_polygon_area_m2:
                continue

        kind = type_name.lower()
        emitted += 1
        yield SkiSiteRecord(
            osm_kind=kind,
            osm_id=int(obj.id),
            geometry=geom,
            tags=tags,
        )


SampleMode = Literal["reservoir", "prefix"]


def _pbf_scan_heartbeat(stop: threading.Event, interval_s: float) -> None:
    """Print elapsed time periodically until stop is set (daemon thread)."""
    t0 = time.monotonic()
    while not stop.wait(timeout=interval_s):
        dt = time.monotonic() - t0
        print(
            f"  ... still scanning PBF ({dt:.0f}s elapsed; large extracts spend a long time "
            f"in the node section before polygon areas) ...",
            flush=True,
        )


def collect_ski_sites(
    pbf_path: str,
    max_samples: int | None,
    bbox: tuple[float, float, float, float] | None,
    min_polygon_area_m2: float,
    random_seed: int,
    sample_mode: SampleMode = "reservoir",
    areas_only: bool = False,
    progress_every: int = 0,
) -> list[SkiSiteRecord]:
    """Materialize ski sites from PBF, optionally subsampling for memory/runtime.

    ``sample_mode``:
        ``reservoir`` — unbiased uniform subsample over the whole file; requires a full PBF scan.
        ``prefix`` — take the first ``max_samples`` accepted records then stop (fast; biased).
    """
    stream = iter_ski_sites_from_pbf(
        pbf_path,
        bbox=bbox,
        min_polygon_area_m2=min_polygon_area_m2,
        areas_only=areas_only,
        progress_every=progress_every,
    )
    if max_samples is None:
        return list(stream)

    if sample_mode == "prefix":
        out: list[SkiSiteRecord] = []
        for row in stream:
            out.append(row)
            if len(out) >= max_samples:
                break
        return out

    if sample_mode != "reservoir":
        raise ValueError(f"Unknown sample_mode: {sample_mode!r}")

    rng = random.Random(random_seed)
    pool: list[SkiSiteRecord] = []
    for i, row in enumerate(stream):
        if len(pool) < max_samples:
            pool.append(row)
        else:
            j = rng.randint(0, i)
            if j < max_samples:
                pool[j] = row
    return pool


def _get_existing_split(ds_path: UPath, group: str, window_name: str) -> str | None:
    meta_path = ds_path / "windows" / group / window_name / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
        s = data.get("options", {}).get("split")
        if s in ("train", "val"):
            return s
    except (json.JSONDecodeError, OSError, TypeError):
        pass
    return None


def _vector_label_complete(dataset: Dataset, group: str, window_name: str) -> bool:
    return dataset.storage.is_layer_completed(group, window_name, LABEL_LAYER, 0)


def _split_from_id(window_key: str, val_fraction: float) -> str:
    """Stable pseudo-random split from a string key."""
    h = int(hashlib.sha256(window_key.encode()).hexdigest(), 16)
    if (h % 10_000) / 10_000.0 < val_fraction:
        return "val"
    return "train"


def _parse_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def create_ski_window_job(
    record: SkiSiteRecord,
    dataset: Dataset,
    ds_path: UPath,
    group: str,
    time_start: datetime,
    time_end: datetime,
    val_fraction: float,
    skip_existing: bool,
) -> None:
    """Create one window labeled ``no_landslide`` centered on the ski feature."""
    pt = _representative_point(record.geometry)
    longitude, latitude = float(pt.x), float(pt.y)

    window_name = f"ski_{record.osm_kind}_{record.osm_id}_{latitude:.4f}_{longitude:.4f}"
    window_key = f"{record.osm_kind}:{record.osm_id}"

    if skip_existing and _vector_label_complete(dataset, group, window_name):
        return

    split = _get_existing_split(ds_path, group, window_name)
    if split is None:
        split = _split_from_id(window_key, val_fraction)

    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)

    bounds = calculate_bounds(dst_geometry, WINDOW_SIZE_PIXELS)

    window = Window(
        storage=dataset.storage,
        group=group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(time_start, time_end),
        options={
            "split": split,
            "latitude": latitude,
            "longitude": longitude,
            "window_size": WINDOW_SIZE_PIXELS,
            "window_type": "osm_ski_false_positive",
            "time_range_start": time_start.isoformat(),
            "time_range_end": time_end.isoformat(),
            "osm_kind": record.osm_kind,
            "osm_id": record.osm_id,
            "piste:type": record.tags.get("piste:type", ""),
            "landuse": record.tags.get("landuse", ""),
            "aerialway": record.tags.get("aerialway", ""),
        },
    )
    window.save()

    features = [
        Feature(
            window.get_geometry(),
            {
                "label": "no_landslide",
                "source": "osm_ski",
                "osm_kind": record.osm_kind,
                "osm_id": str(record.osm_id),
            },
        ),
    ]
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(layer_dir, features)
    window.mark_layer_completed(LABEL_LAYER)


def create_windows_from_pbf(
    pbf_path: UPath,
    ds_path: UPath,
    group: str = DEFAULT_GROUP,
    max_samples: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    min_polygon_area_m2: float = 0.0,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    val_fraction: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 32,
    skip_existing: bool = True,
    sample_mode: SampleMode = "reservoir",
    areas_only: bool = False,
    progress_every: int = 0,
    progress_seconds: float = 0.0,
) -> None:
    """Scan PBF for ski features and write rslearn windows under ``ds_path``."""
    if time_start is None or time_end is None:
        time_start = datetime(2019, 6, 1, tzinfo=timezone.utc)
        time_end = datetime(2019, 8, 31, tzinfo=timezone.utc)

    print(f"Reading ski features from {pbf_path} (sample_mode={sample_mode}) ...")
    if sample_mode == "prefix" and areas_only:
        print(
            "Hint: on large extracts, nodes precede ways in the PBF; expect a long read "
            "before the first polygon Area, or clip with osmium extract (see script docstring).",
            flush=True,
        )

    hb_stop: threading.Event | None = None
    hb_thread: threading.Thread | None = None
    if progress_seconds > 0:
        hb_stop = threading.Event()
        hb_thread = threading.Thread(
            target=_pbf_scan_heartbeat,
            args=(hb_stop, progress_seconds),
            daemon=True,
        )
        hb_thread.start()

    try:
        records = collect_ski_sites(
            str(pbf_path),
            max_samples=max_samples,
            bbox=bbox,
            min_polygon_area_m2=min_polygon_area_m2,
            random_seed=random_seed,
            sample_mode=sample_mode,
            areas_only=areas_only,
            progress_every=progress_every,
        )
    finally:
        if hb_stop is not None:
            hb_stop.set()
        if hb_thread is not None:
            hb_thread.join(timeout=1.0)
    print(f"Collected {len(records)} ski sites for windows.")

    dataset = Dataset(ds_path)
    jobs = [
        dict(
            record=r,
            dataset=dataset,
            ds_path=ds_path,
            group=group,
            time_start=time_start,
            time_end=time_end,
            val_fraction=val_fraction,
            skip_existing=skip_existing,
        )
        for r in records
    ]

    if skip_existing:
        print("skip_existing: windows with completed vector 'label' layer are skipped (--force to rebuild)")

    p = multiprocessing.Pool(num_workers)
    outputs = star_imap_unordered(p, create_ski_window_job, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(
        description="Create rslearn windows from OSM PBF ski features (false positives for landslide detection)",
        epilog="Large extracts: use a regional PBF or `osmium extract -b minlon,minlat,maxlon,maxlat` "
        "before running; continent files list billions of nodes before polygon areas appear. "
        "Install the extract CLI with: conda install -c conda-forge osmium-tool (not pip).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pbf_path",
        type=str,
        required=True,
        help="Path to an OSM PBF (e.g. regional extract; continent-scale is slow and memory-heavy)",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Output rslearn dataset directory",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=DEFAULT_GROUP,
        help=f"Window group name (default: {DEFAULT_GROUP})",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap how many sites to keep (default: keep all). With --sample_mode reservoir this "
        "implies a full PBF scan; with prefix, reading stops once this many are collected.",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        choices=("reservoir", "prefix"),
        default="reservoir",
        help="reservoir: unbiased random sample over the entire file (slow on continent PBFs). "
        "prefix: stop after N accepted sites (still slow on full Europe + --areas_only until the "
        "ways section; clip the PBF first).",
    )
    parser.add_argument(
        "--areas_only",
        action="store_true",
        help="Only use polygon Area features (skip nodes and line ways); pairs well with "
        "prefix sampling for ski-resort footprints.",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
        help="Optional geographic filter on feature representative point",
    )
    parser.add_argument(
        "--min_polygon_area_m2",
        type=float,
        default=0.0,
        help="Minimum polygon area (m^2) for Area features; 0 disables (default: 0)",
    )
    parser.add_argument(
        "--time_start",
        type=str,
        default="2019-06-01",
        help="Window time range start (ISO-8601 date or datetime, UTC if naive)",
    )
    parser.add_argument(
        "--time_end",
        type=str,
        default="2019-08-31",
        help="Window time range end (ISO-8601)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
        help="Fraction of windows assigned to val via stable hash (default: 0.15)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="RNG seed for --sample_mode reservoir (ignored for prefix)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Worker processes for window writes (default: 32)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild windows even if the vector label layer is already completed",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=0,
        metavar="N",
        help="Print heartbeat every N key-filtered OSM objects during the scan (0=off). "
        "Use e.g. 5000000 on huge files so long node-section reads do not look stuck.",
    )
    parser.add_argument(
        "--progress_seconds",
        type=float,
        default=120.0,
        metavar="SEC",
        help="Wall-clock heartbeat every SEC seconds while scanning the PBF (0=off). "
        "Default 120 so silent node-section reads on Europe-sized files do not look hung.",
    )
    args = parser.parse_args()

    create_windows_from_pbf(
        UPath(args.pbf_path),
        UPath(args.ds_path),
        group=args.group,
        max_samples=args.max_samples,
        bbox=tuple(args.bbox) if args.bbox is not None else None,
        min_polygon_area_m2=args.min_polygon_area_m2,
        time_start=_parse_time(args.time_start),
        time_end=_parse_time(args.time_end),
        val_fraction=args.val_fraction,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        skip_existing=not args.force,
        sample_mode=cast(SampleMode, args.sample_mode),
        areas_only=args.areas_only,
        progress_every=args.progress_every,
        progress_seconds=args.progress_seconds,
    )
