"""Export a year's per-block provenance JSON as GeoParquet tables for release.

Three tables, each under ``<table>/year=<year>/`` so a reader can load one year:

``windows``
    One row per 2048 px window: the block and code that wrote it, its context margin,
    whether it was written, and its footprint in WGS84.
``inputs``
    One row per scene per monthly mosaic per sensor per window, in mosaic order.
    No geometry; join to ``windows`` on (crs, window) and to ``scenes`` on scene_id.
``scenes``
    One row per scene used that year, with its capture time and WGS84 footprint, so
    a footprint is stored once however many windows used it.

Only blocks with a completion marker are exported, since a block being recomputed may
have a provenance file newer than its embeddings. Bounds are in metres in the zone's
northern CRS (EPSG:326NN), whose northing continues below zero south of the equator.
"""

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet as pq
import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from upath import UPath

from rslp.large_scale_embeddings.predict_pipeline import (
    COMPLETED_DIR_PREFIX,
    PATCH_SIZE,
    PROVENANCE_DIR_PREFIX,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)

GEOPARQUET_VERSION = "1.1.0"
COMPRESSION = "zstd"

TIMESTAMP = pa.timestamp("us", tz="UTC")
BBOX = pa.struct(
    [
        ("xmin", pa.float64()),
        ("ymin", pa.float64()),
        ("xmax", pa.float64()),
        ("ymax", pa.float64()),
    ]
)

WINDOWS_SCHEMA = pa.schema(
    [
        ("crs", pa.string()),
        ("block", pa.string()),
        ("window", pa.string()),
        ("xmin", pa.float64()),
        ("ymin", pa.float64()),
        ("xmax", pa.float64()),
        ("ymax", pa.float64()),
        ("context_pixels", pa.int16()),
        ("written", pa.bool_()),
        ("window_size", pa.int16()),
        ("worker", pa.string()),
        ("code_version", pa.string()),
        ("geometry", pa.binary()),
        ("bbox", BBOX),
    ]
)
INPUTS_SCHEMA = pa.schema(
    [
        ("crs", pa.string()),
        ("window", pa.string()),
        ("sensor", pa.string()),
        ("period_start", TIMESTAMP),
        ("period_end", TIMESTAMP),
        ("rank", pa.int8()),
        ("scene_id", pa.string()),
    ]
)
SCENES_SCHEMA = pa.schema(
    [
        ("scene_id", pa.string()),
        ("sensor", pa.string()),
        ("acquired", TIMESTAMP),
        ("geometry", pa.binary()),
        ("bbox", BBOX),
    ]
)


def _timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _wgs84(geometry: STGeometry) -> shapely.Geometry:
    if geometry.projection.crs != WGS84_PROJECTION.crs:
        geometry = geometry.to_projection(WGS84_PROJECTION)
    return geometry.shp


def _bbox(shp: shapely.Geometry) -> dict[str, float]:
    xmin, ymin, xmax, ymax = shp.bounds
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def _block_rows(
    marker: dict, provenance: dict, block: str
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Flatten one block's marker and provenance into table rows.

    Args:
        marker: the block's completion marker.
        provenance: the block's provenance file.
        block: the block name, the marker's basename without ".json".

    Returns:
        the block's windows rows, inputs rows, and scenes keyed by scene_id.

    Raises:
        ValueError: if a window's recorded bounds are not its crop plus the margin.
    """
    crs = provenance["projection"]["crs"]
    projection = Projection(
        CRS.from_string(crs),
        provenance["projection"]["x_resolution"],
        provenance["projection"]["y_resolution"],
    )
    # Absent in files written before margins existed, which means 0.
    context_pixels = provenance.get("context_pixels", 0)
    recorded_bounds = provenance.get("window_bounds", {})
    written = {tuple(origin) for origin in marker.get("written", [])}

    windows: list[dict] = []
    inputs: list[dict] = []
    scenes: dict[str, dict] = {}
    for name, layers in provenance["windows"].items():
        col, row = (int(v) for v in name.split("_"))
        core = (
            col * PATCH_SIZE,
            row * PATCH_SIZE,
            (col + 1) * PATCH_SIZE,
            (row + 1) * PATCH_SIZE,
        )
        padded = [
            core[0] - context_pixels,
            core[1] - context_pixels,
            core[2] + context_pixels,
            core[3] + context_pixels,
        ]
        if name in recorded_bounds and recorded_bounds[name] != padded:
            raise ValueError(
                f"{block} window {name} has bounds {recorded_bounds[name]}, "
                f"not its crop {core} plus {context_pixels} px"
            )
        footprint = _wgs84(STGeometry(projection, shapely.box(*core), None))
        # Pixel rows grow southward, so northing is the negated pixel y.
        x_res, y_res = projection.x_resolution, projection.y_resolution
        windows.append(
            {
                "crs": crs,
                "block": block,
                "window": name,
                "xmin": core[0] * x_res,
                "ymin": core[3] * y_res,
                "xmax": core[2] * x_res,
                "ymax": core[1] * y_res,
                "context_pixels": context_pixels,
                "written": (core[0], core[1]) in written,
                "window_size": marker.get("window_size"),
                "worker": marker.get("worker"),
                "code_version": provenance.get("code_version"),
                "geometry": shapely.to_wkb(footprint),
                "bbox": _bbox(footprint),
            }
        )
        for sensor, groups in layers.items():
            for group in groups:
                period = group["time_range"] or [None, None]
                for rank, item in enumerate(group["items"]):
                    scene_id = item["name"]
                    inputs.append(
                        {
                            "crs": crs,
                            "window": name,
                            "sensor": sensor,
                            "period_start": _timestamp(period[0]),
                            "period_end": _timestamp(period[1]),
                            "rank": rank,
                            "scene_id": scene_id,
                        }
                    )
                    if scene_id not in scenes:
                        geometry = STGeometry.deserialize(item["geometry"])
                        shp = _wgs84(geometry)
                        scenes[scene_id] = {
                            "scene_id": scene_id,
                            "sensor": sensor,
                            "acquired": (
                                geometry.time_range[0]
                                if geometry.time_range is not None
                                else None
                            ),
                            "geometry": shapely.to_wkb(shp),
                            "bbox": _bbox(shp),
                        }
    return windows, inputs, scenes


def _geo_metadata(table: pa.Table) -> bytes:
    """GeoParquet metadata for a WKB geometry column in OGC:CRS84 with a bbox covering."""
    shapes = shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
    types = sorted({shape.geom_type for shape in shapes})
    if len(shapes):
        xmin, ymin, xmax, ymax = shapely.total_bounds(shapes).tolist()
        bounds = [xmin, ymin, xmax, ymax]
    else:
        bounds = []
    column: dict[str, Any] = {
        "encoding": "WKB",
        "geometry_types": types,
        # No "crs" key means OGC:CRS84, longitude then latitude.
        "covering": {
            "bbox": {key: ["bbox", key] for key in ("xmin", "ymin", "xmax", "ymax")}
        },
    }
    if bounds:
        column["bbox"] = bounds
    return json.dumps(
        {
            "version": GEOPARQUET_VERSION,
            "primary_column": "geometry",
            "columns": {"geometry": column},
        }
    ).encode()


def _write_table(
    rows: list[dict], schema: pa.Schema, filesystem: Any, path: str, geo: bool
) -> int:
    table = pa.Table.from_pylist(rows, schema=schema)
    if geo:
        table = table.replace_schema_metadata({b"geo": _geo_metadata(table)})
    filesystem.create_dir(path.rsplit("/", 1)[0], recursive=True)
    pq.write_table(table, path, filesystem=filesystem, compression=COMPRESSION)
    return table.num_rows


def _batches(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def export_provenance(
    run_path: str,
    year: int,
    out_dir: str,
    workers: int = 64,
    batch_blocks: int = 1000,
) -> None:
    """Export one year's provenance to windows, inputs and scenes GeoParquet tables.

    Args:
        run_path: the directory holding completed_<year>/ and provenance_<year>/.
        year: the reference year to export.
        out_dir: where to write the tables, a local path or a pyarrow filesystem URI.
        workers: threads reading marker and provenance files.
        batch_blocks: blocks per row group of the inputs table, which bounds memory.
    """
    completed = UPath(run_path) / f"{COMPLETED_DIR_PREFIX}_{year}"
    provenance_dir = UPath(run_path) / f"{PROVENANCE_DIR_PREFIX}_{year}"
    markers = sorted(p.name for p in completed.iterdir() if p.name.endswith(".json"))
    provenance_names = {p.name for p in provenance_dir.iterdir()}
    missing = [name for name in markers if name not in provenance_names]
    if missing:
        logger.warning(
            "%d of %d blocks have a marker but no provenance, e.g. %s",
            len(missing),
            len(markers),
            missing[0],
        )
    blocks = [name for name in markers if name in provenance_names]
    logger.info("exporting %d blocks of %d from %s", len(blocks), year, run_path)

    vanished: list[str] = []

    def read(name: str) -> tuple[list[dict], list[dict], dict[str, dict]]:
        try:
            marker = json.loads((completed / name).read_text())
            provenance = json.loads((provenance_dir / name).read_text())
        except FileNotFoundError:
            # Deleted since the listing, to be recomputed; its rows come next export.
            vanished.append(name)
            return [], [], {}
        return _block_rows(marker, provenance, name.removesuffix(".json"))

    filesystem, root = pyarrow.fs.FileSystem.from_uri(out_dir)
    partition = f"year={year}"
    inputs_path = f"{root}/inputs/{partition}/part-0.parquet"
    filesystem.create_dir(inputs_path.rsplit("/", 1)[0], recursive=True)

    windows: list[dict] = []
    scenes: dict[str, dict] = {}
    n_inputs = 0
    with (
        ThreadPoolExecutor(workers) as pool,
        pq.ParquetWriter(
            inputs_path, INPUTS_SCHEMA, filesystem=filesystem, compression=COMPRESSION
        ) as inputs_writer,
    ):
        for i, batch in enumerate(_batches(blocks, batch_blocks)):
            inputs: list[dict] = []
            for block_windows, block_inputs, block_scenes in pool.map(read, batch):
                windows += block_windows
                inputs += block_inputs
                for scene_id, scene in block_scenes.items():
                    scenes.setdefault(scene_id, scene)
            inputs_writer.write_table(
                pa.Table.from_pylist(inputs, schema=INPUTS_SCHEMA)
            )
            n_inputs += len(inputs)
            logger.info(
                "%d of %d blocks read, %d input rows, %d scenes",
                min((i + 1) * batch_blocks, len(blocks)),
                len(blocks),
                n_inputs,
                len(scenes),
            )

    n_windows = _write_table(
        windows,
        WINDOWS_SCHEMA,
        filesystem,
        f"{root}/windows/{partition}/part-0.parquet",
        geo=True,
    )
    if vanished:
        logger.warning(
            "%d blocks were deleted while exporting and are left out, e.g. %s",
            len(vanished),
            vanished[0],
        )
    n_scenes = _write_table(
        sorted(scenes.values(), key=lambda scene: scene["scene_id"]),
        SCENES_SCHEMA,
        filesystem,
        f"{root}/scenes/{partition}/part-0.parquet",
        geo=True,
    )
    logger.info(
        "wrote %d windows, %d input rows and %d scenes for %d to %s",
        n_windows,
        n_inputs,
        n_scenes,
        year,
        out_dir,
    )
