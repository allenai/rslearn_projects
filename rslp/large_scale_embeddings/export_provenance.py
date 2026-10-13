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
    a footprint is stored once however many windows used it. Scene IDs are our
    datasets service's, so each scene is also resolved to the public data it came
    from: the providers our dataset config allows for its sensor, each with its own
    item ID, collection and asset location, and the provider used when only one is
    allowed. A reader never needs access to the datasets service.

Only blocks with a completion marker are exported, since a block being recomputed may
have a provenance file newer than its embeddings. Bounds are in metres in the zone's
northern CRS (EPSG:326NN), whose northing continues below zero south of the equator.
"""

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet as pq
import requests
import shapely
from rasterio.crs import CRS
from requests.adapters import HTTPAdapter
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from upath import UPath
from urllib3.util.retry import Retry

from rslp.large_scale_embeddings.predict_pipeline import (
    COMPLETED_DIR_PREFIX,
    DATASET_CONFIG_FNAME,
    PATCH_SIZE,
    PROVENANCE_DIR_PREFIX,
    EmbeddingInputs,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)

GEOPARQUET_VERSION = "1.1.0"
COMPRESSION = "zstd"

# The datasets service the scene IDs belong to, configured as for the workers.
DATASETS_API_URL_ENV_VAR = "OEDATASETS_API_URL"
DATASETS_API_TOKEN_ENV_VAR = "DATASETS_API_TOKEN"  # nosec
DEFAULT_DATASETS_API_URL = "https://datasets.olmoearth.allenai.org"
LOOKUP_BATCH = 500
LOOKUP_WORKERS = 8

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
SOURCE = pa.struct(
    [
        ("provider", pa.string()),
        ("collection", pa.string()),
        ("item_id", pa.string()),
        ("asset_prefix", pa.string()),
    ]
)
SCENES_SCHEMA = pa.schema(
    [
        ("scene_id", pa.string()),
        ("sensor", pa.string()),
        ("acquired", TIMESTAMP),
        ("product_id", pa.string()),
        ("cloud_cover", pa.float64()),
        ("provider", pa.string()),
        ("sources", pa.list_(SOURCE)),
        ("resolved", pa.bool_()),
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


def provider_excludes(dataset_config_fname: str) -> dict[str, set[str]]:
    """The providers each layer of a dataset config excludes, by layer name.

    The config names providers by enum member (AWS_USGS_LANDSAT) and the service keys
    them by value (aws_usgs_landsat), so names are lowercased to match the service.
    """
    with UPath(dataset_config_fname).open() as f:
        layers = json.load(f)["layers"]
    return {
        name: {
            provider.lower()
            for provider in layer.get("data_source", {})
            .get("init_args", {})
            .get("provider_excludes")
            or []
        }
        for name, layer in layers.items()
    }


def _asset_prefix(urls: list[str]) -> str | None:
    """The deepest directory holding every asset URL."""
    if not urls:
        return None
    return os.path.commonprefix(urls).rsplit("/", 1)[0] + "/"


def _lookup_records(
    scene_ids: list[str], api_url: str, token: str, workers: int = LOOKUP_WORKERS
) -> dict[str, dict]:
    """Fetch the datasets service record of each scene ID that it still has."""
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=workers))
    url = f"{api_url.rstrip('/')}/api/v1/items/search"

    def fetch(batch: list[str]) -> list[dict]:
        response = session.post(
            url,
            json={
                "id": {"inc": batch},
                "limit": len(batch),
                "offset": 0,
                "sign_urls": False,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["records"]

    records: dict[str, dict] = {}
    with ThreadPoolExecutor(workers) as pool:
        for i, batch_records in enumerate(
            pool.map(fetch, list(_batches(scene_ids, LOOKUP_BATCH)))
        ):
            records.update((record["id"], record) for record in batch_records)
            if i % 100 == 0:
                logger.info("resolved %d of %d scenes", len(records), len(scene_ids))
    return records


def _resolve(scene: dict, record: dict | None, excludes: set[str]) -> dict:
    """Add a scene's public sources from its datasets service record."""
    if record is None:
        return {
            **scene,
            "product_id": None,
            "cloud_cover": None,
            "provider": None,
            "sources": [],
            "resolved": False,
        }
    sources = [
        {
            "provider": name,
            "collection": provider.get("collection"),
            "item_id": provider["id"],
            "asset_prefix": _asset_prefix(
                [
                    asset["url"]
                    for asset in provider.get("assets", [])
                    if asset.get("url")
                ]
            ),
        }
        for name, provider in sorted(record["data_providers"].items())
        if name not in excludes
    ]
    properties = record.get("properties", {})
    # Sentinel-2 records carry the ESA product name; otherwise the providers' own ID,
    # if they agree on one (the USGS mirrors do).
    item_ids = {source["item_id"] for source in sources}
    product_id = properties.get("product_id") or (
        item_ids.pop() if len(item_ids) == 1 else None
    )
    return {
        **scene,
        "product_id": product_id,
        "cloud_cover": properties.get("cloud_cover"),
        # With several allowed, the reader picked one at random per read.
        "provider": sources[0]["provider"] if len(sources) == 1 else None,
        "sources": sources,
        "resolved": True,
    }


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
    inputs: EmbeddingInputs = EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
    resolve_sources: bool = True,
) -> None:
    """Export one year's provenance to windows, inputs and scenes GeoParquet tables.

    Resolving sources reads the datasets service's URL and token from the same
    environment variables the workers use (OEDATASETS_API_URL, DATASETS_API_TOKEN).

    Args:
        run_path: the directory holding completed_<year>/ and provenance_<year>/.
        year: the reference year to export.
        out_dir: where to write the tables, a local path or a pyarrow filesystem URI.
        workers: threads reading marker and provenance files.
        batch_blocks: blocks per row group of the inputs table, which bounds memory.
        inputs: the input variant the run used, whose dataset config names the
            providers each sensor excluded.
        resolve_sources: look each scene up in the datasets service to record the
            public data it came from. Off leaves those columns empty.

    Raises:
        ValueError: if resolving sources without a datasets service token.
    """
    token = os.environ.get(DATASETS_API_TOKEN_ENV_VAR)
    if resolve_sources and not token:
        raise ValueError(
            f"set {DATASETS_API_TOKEN_ENV_VAR} to resolve sources, or pass "
            "resolve_sources false"
        )
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
            input_rows: list[dict] = []
            for block_windows, block_inputs, block_scenes in pool.map(read, batch):
                windows += block_windows
                input_rows += block_inputs
                for scene_id, scene in block_scenes.items():
                    scenes.setdefault(scene_id, scene)
            inputs_writer.write_table(
                pa.Table.from_pylist(input_rows, schema=INPUTS_SCHEMA)
            )
            n_inputs += len(input_rows)
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
    records: dict[str, dict] = {}
    if resolve_sources and token:
        api_url = os.environ.get(DATASETS_API_URL_ENV_VAR, DEFAULT_DATASETS_API_URL)
        records = _lookup_records(sorted(scenes), api_url, token)
        unresolved = len(scenes) - len(records.keys() & scenes.keys())
        if unresolved:
            logger.warning(
                "%d of %d scenes are no longer in the datasets service",
                unresolved,
                len(scenes),
            )
    excludes = provider_excludes(DATASET_CONFIG_FNAME.format(inputs=inputs.value))
    resolved = [
        _resolve(scene, records.get(scene_id), excludes.get(scene["sensor"], set()))
        for scene_id, scene in sorted(scenes.items())
    ]
    n_scenes = _write_table(
        resolved,
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
