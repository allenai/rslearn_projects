"""Export a year's per-block provenance JSON as GeoParquet tables for release.

Three tables, each written as ``<table>/<year>.parquet`` with a ``year`` column, so a
file read on its own still says which year it holds and a glob over all years needs
no partitioning scheme:

``windows``
    One row per 2048 px (20.48 km) window of one UTM zone and year. Each window is
    exactly one shard of the embeddings zarr. Holds the block that wrote it, its
    context margin, whether it was written, and its footprint in WGS84. Rows follow a
    Hilbert curve in small row groups, so a bbox query against a remote file reads a
    few of them.
``window_scenes``
    Which scenes went into each window: one row per scene per monthly mosaic per
    modality per window. ``mosaic_order`` 0 is the scene that supplies most pixels;
    later scenes only fill pixels the earlier ones left empty, since the run
    composites by first valid pixel. No geometry; join to ``windows`` on window_id
    and to ``scenes`` on scene_id. Sorted by (zone, row, col), so a lookup of one
    window reads one row group.
``scenes``
    One row per scene used that year, keyed by ``scene_id``, the scene's item ID in
    our datasets service. Other columns follow the service's names: ``data_provider``,
    ``collection`` and ``vendor_id`` name the one provider our dataset config allows
    for the scene's modality, with that provider's own ID for it. ``stac_item_url``
    links the scene in a public STAC catalog, which lists its files, so a reader
    never needs access to the datasets service. The WGS84 footprint is stored once
    however many windows used the scene. Sorted by scene_id in small row groups, so
    looking up a window's scenes by ID reads a few of them.

A window is keyed by its shard in the zarr: ``zone`` names the zero-padded ``utm<NN>``
group (utm02 for zone 2), and ``col`` and ``row`` are the shard's chunk indices there,
joined as ``window_id`` "<zone>_<col>_<row>".

Only blocks with a completion marker are exported, since a block being recomputed may
have a provenance file newer than its embeddings. UTM bounds are in metres in the
zone's northern CRS (EPSG:326NN), whose northing continues below zero south of the
equator. Bbox coverings are float32, rounded outward so they still contain the
geometry.
"""

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet as pq
import requests
import shapely
import shapely.affinity
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
from rslp.large_scale_embeddings.render_web_pca import (
    UTM_NORTH_EPSG_BASE,
    UTM_ORIGIN_X,
    UTM_ORIGIN_Y,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)

GEOPARQUET_VERSION = "1.1.0"
COMPRESSION = "zstd"
# Rows per row group: small for the geometry tables so a remote reader skips most of
# them by bbox or scene_id, larger for window_scenes where one window is ~50 rows.
WINDOWS_ROW_GROUP = 10_000
SCENES_ROW_GROUP = 10_000
WINDOW_SCENES_ROW_GROUP = 100_000
WINDOW_SCENES_SORT = [
    ("zone", "ascending"),
    ("row", "ascending"),
    ("col", "ascending"),
    ("modality", "ascending"),
    ("period_start", "ascending"),
    ("mosaic_order", "ascending"),
]
HILBERT_ORDER = 16
WORLD = shapely.box(-180, -90, 180, 90)

# The datasets service the scene IDs belong to, configured as for the workers.
DATASETS_API_URL_ENV_VAR = "OEDATASETS_API_URL"
DATASETS_API_TOKEN_ENV_VAR = "DATASETS_API_TOKEN"  # nosec
DEFAULT_DATASETS_API_URL = "https://datasets.olmoearth.allenai.org"
LOOKUP_BATCH = 500
# Public STAC item URLs by data provider. Planetary Computer's collection IDs match
# the service's; USGS-on-GCS Landsat Level-1 is catalogued by USGS as landsat-c2l1.
PLANETARY_COMPUTER_ITEM_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/"
    "{collection}/items/{vendor_id}"
)
STAC_ITEM_URLS = {
    ("gcs_usgs_landsat", "landsat-8-9-c2-l1"): (
        "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/"
        "{vendor_id}"
    ),
}
LOOKUP_WORKERS = 8

TIMESTAMP = pa.timestamp("us", tz="UTC")
BBOX = pa.struct(
    [
        ("xmin", pa.float32()),
        ("ymin", pa.float32()),
        ("xmax", pa.float32()),
        ("ymax", pa.float32()),
    ]
)


def _schema(fields: list[tuple[str, pa.DataType, str]]) -> pa.Schema:
    """A schema whose fields carry their descriptions as Arrow field metadata."""
    return pa.schema(
        [
            pa.field(name, dtype, metadata={"description": description})
            for name, dtype, description in fields
        ]
    )


YEAR_DESCRIPTION = "The reference year of the embeddings this row belongs to."
BBOX_DESCRIPTION = (
    "The footprint's bounding box in WGS84, rounded outward, which readers use to "
    "skip row groups."
)
WINDOWS_SCHEMA = _schema(
    [
        ("year", pa.int32(), YEAR_DESCRIPTION),
        (
            "window_id",
            pa.string(),
            'The window\'s key, "<zone>_<col>_<row>", which joins it to window_scenes.',
        ),
        (
            "zone",
            pa.int16(),
            "The UTM zone, whose zarr group is the zero-padded utm<NN>.",
        ),
        (
            "col",
            pa.int32(),
            "The window's shard index along x in utm<NN>/embeddings, counting east "
            "from 0 m easting in steps of 20,480 m.",
        ),
        (
            "row",
            pa.int32(),
            "The window's shard index along y in utm<NN>/embeddings, counting south "
            "from 9,502,720 m northing in steps of 20,480 m.",
        ),
        (
            "crs",
            pa.string(),
            "The zone's northern UTM CRS (EPSG:326NN) that the UTM bounds are given "
            "in.",
        ),
        (
            "block",
            pa.string(),
            "The name of the processing block that wrote this window, which links it "
            "to the run's internal records.",
        ),
        ("utm_xmin", pa.int32(), "The window's western edge in metres of easting."),
        ("utm_ymin", pa.int32(), "The window's southern edge in metres of northing."),
        ("utm_xmax", pa.int32(), "The window's eastern edge in metres of easting."),
        ("utm_ymax", pa.int32(), "The window's northern edge in metres of northing."),
        (
            "context_pixels",
            pa.int16(),
            "The margin of extra pixels read around the window at inference (0 or "
            "48), so its edges are computed with surrounding context.",
        ),
        (
            "written",
            pa.bool_(),
            "Whether the window has embeddings; false when it had no Sentinel-2 "
            "imagery, which the model requires.",
        ),
        (
            "inference_crop_size",
            pa.int16(),
            "Size in pixels of the crops the model ran on within the window (480 on "
            "H100s, 384 on A100s). Crops overlap by 32 px and are merged. Null where "
            "the run didn't record it.",
        ),
        ("geometry", pa.binary(), "The window's footprint in WGS84."),
        ("bbox", BBOX, BBOX_DESCRIPTION),
    ]
)
WINDOW_SCENES_SCHEMA = _schema(
    [
        ("year", pa.int32(), YEAR_DESCRIPTION),
        (
            "window_id",
            pa.string(),
            "The window that used this scene, which joins to windows.",
        ),
        (
            "zone",
            pa.int16(),
            "The window's UTM zone, repeated so readers can find a window's rows from "
            "row-group statistics.",
        ),
        ("col", pa.int32(), "The window's shard column, repeated for the same reason."),
        ("row", pa.int32(), "The window's shard row, repeated for the same reason."),
        (
            "modality",
            pa.string(),
            "The model input this scene filled: sentinel2_l2a, sentinel1 or landsat.",
        ),
        (
            "period_start",
            pa.date32(),
            "The first day of the 30-day period whose mosaic this scene went into.",
        ),
        (
            "period_end",
            pa.date32(),
            "The day after that period ends, which is 30 days after period_start.",
        ),
        (
            "mosaic_order",
            pa.int8(),
            "The scene's priority in that period's mosaic: 0 is the main scene, and "
            "higher numbers only fill pixels the earlier scenes left empty.",
        ),
        (
            "scene_id",
            pa.string(),
            "The scene's datasets-service item ID, which joins to scenes.",
        ),
    ]
)
SCENES_SCHEMA = _schema(
    [
        ("year", pa.int32(), YEAR_DESCRIPTION),
        (
            "scene_id",
            pa.string(),
            "The scene's datasets-service item ID, the table's key.",
        ),
        (
            "modality",
            pa.string(),
            "The model input the scene fed: sentinel2_l2a, sentinel1 or landsat.",
        ),
        (
            "satellite",
            pa.string(),
            "The satellite that captured the scene, as the service codes it, such as "
            "S2A, S1C or LC09.",
        ),
        (
            "collected_at",
            TIMESTAMP,
            "The scene's collection time, as the service records it.",
        ),
        (
            "product_id",
            pa.string(),
            "The ESA SAFE product name for Sentinel-2 scenes, which includes the "
            "processing baseline. Null for other modalities, whose vendor_id is "
            "already the product name.",
        ),
        (
            "cloud_cover",
            pa.float64(),
            "The scene's cloud cover percentage, null for Sentinel-1 radar.",
        ),
        (
            "data_provider",
            pa.string(),
            "The service's name for the provider the run read the scene from "
            "(planetary_computer or gcs_usgs_landsat).",
        ),
        (
            "collection",
            pa.string(),
            "The service's collection name for that provider, such as "
            "sentinel-2-l2a.",
        ),
        ("vendor_id", pa.string(), "The data provider's own ID for the scene."),
        (
            "stac_item_url",
            pa.string(),
            "The scene's item in a public STAC catalog, which lists its files: Planetary "
            "Computer, or USGS landsat-c2l1 for Landsat.",
        ),
        (
            "geometry",
            pa.binary(),
            "The scene's footprint in WGS84, split into two parts if it crosses "
            "+/-180 degrees.",
        ),
        ("bbox", BBOX, BBOX_DESCRIPTION),
    ]
)


def _date(value: str | None) -> date | None:
    """The UTC date of a period boundary, which the run always puts at midnight.

    Raises:
        ValueError: if the boundary is not midnight UTC, so a date would lose time.
    """
    if value is None:
        return None
    moment = datetime.fromisoformat(value).astimezone(timezone.utc)
    if moment.time() != datetime.min.time():
        raise ValueError(f"period boundary {value} is not midnight UTC")
    return moment.date()


def _wgs84(geometry: STGeometry) -> shapely.Geometry:
    if geometry.projection.crs != WGS84_PROJECTION.crs:
        geometry = geometry.to_projection(WGS84_PROJECTION)
    return _split_antimeridian(geometry.shp)


def _split_antimeridian(shp: shapely.Geometry) -> shapely.Geometry:
    """Fold the parts of a footprint past ±180° back onto the map as separate polygons.

    Some footprints come from the datasets service unwrapped, running to 180.7°.
    """
    xmin, _, xmax, _ = shp.bounds
    if xmin >= -180 and xmax <= 180:
        return shp
    pieces = [
        shp.intersection(WORLD),
        shapely.affinity.translate(
            shp.intersection(shapely.box(180, -90, 540, 90)), -360
        ),
        shapely.affinity.translate(
            shp.intersection(shapely.box(-540, -90, -180, 90)), 360
        ),
    ]
    polygons = [
        part
        for piece in pieces
        for part in shapely.get_parts(piece)
        if isinstance(part, shapely.Polygon) and not part.is_empty
    ]
    return polygons[0] if len(polygons) == 1 else shapely.MultiPolygon(polygons)


def _hilbert_order(table: pa.Table) -> np.ndarray:
    """Row order along a Hilbert curve through the bbox centres.

    Rows near each other on the map then share row groups.
    """
    bbox = table.column("bbox").combine_chunks()
    x = (bbox.field("xmin").to_numpy() + bbox.field("xmax").to_numpy()) / 2
    y = (bbox.field("ymin").to_numpy() + bbox.field("ymax").to_numpy()) / 2
    n = 1 << HILBERT_ORDER
    xi = np.clip(((x + 180) / 360 * n).astype(np.int64), 0, n - 1)
    yi = np.clip(((y + 90) / 180 * n).astype(np.int64), 0, n - 1)
    d = np.zeros_like(xi)
    s = n >> 1
    while s > 0:
        rx = (xi & s) > 0
        ry = (yi & s) > 0
        d += s * s * ((3 * rx.astype(np.int64)) ^ ry.astype(np.int64))
        flip = ~ry & rx
        xi = np.where(flip, n - 1 - xi, xi)
        yi = np.where(flip, n - 1 - yi, yi)
        xi, yi = np.where(ry, xi, yi), np.where(ry, yi, xi)
        s >>= 1
    return np.argsort(d, kind="stable")


def _window_key(projection: Projection, col: int, row: int) -> tuple[int, int, int]:
    """The zone and zarr shard indices of the window at (col, row) in 2048 px units.

    Raises:
        ValueError: if the projection is not a northern UTM zone on the store's grid.
    """
    zone = (CRS.from_user_input(projection.crs).to_epsg() or 0) - UTM_NORTH_EPSG_BASE
    if not 1 <= zone <= 60:
        raise ValueError(f"{projection.crs} is not a northern UTM zone")
    x_span = PATCH_SIZE * projection.x_resolution
    y_span = PATCH_SIZE * -projection.y_resolution
    # Pixel rows grow southward, so a window's northern edge is -row * y_span.
    shard_col, x_off = divmod(col * x_span - UTM_ORIGIN_X, x_span)
    shard_row, y_off = divmod(UTM_ORIGIN_Y + row * y_span, y_span)
    if x_off or y_off:
        raise ValueError(f"window {col}_{row} is off the zarr shard grid")
    return zone, int(shard_col), int(shard_row)


def _bbox(shp: shapely.Geometry) -> dict[str, float]:
    """The geometry's bounds in float32, each rounded away from the geometry."""
    xmin, ymin, xmax, ymax = shp.bounds

    def outward(value: float, toward: float) -> float:
        rounded = np.float32(value)
        # Compare as float64: NumPy 2 casts a Python float to float32 to compare it
        # with a float32, which hides the rounding this checks for.
        if (float(rounded) > value) if toward < 0 else (float(rounded) < value):
            rounded = np.nextafter(rounded, np.float32(toward))
        return float(rounded)

    return {
        "xmin": outward(xmin, -np.inf),
        "ymin": outward(ymin, -np.inf),
        "xmax": outward(xmax, np.inf),
        "ymax": outward(ymax, np.inf),
    }


def _block_rows(
    marker: dict, provenance: dict, block: str, year: int
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Flatten one block's marker and provenance into table rows.

    Args:
        marker: the block's completion marker.
        provenance: the block's provenance file.
        block: the block name, the marker's basename without ".json".
        year: the reference year, stamped on every row.

    Returns:
        the block's windows rows, window_scenes rows, and scenes keyed by scene_id.

    Raises:
        ValueError: if a window's recorded bounds are not its crop plus the margin,
            or it is not on the zarr shard grid.
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
    window_scenes: list[dict] = []
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
        zone, shard_col, shard_row = _window_key(projection, col, row)
        key = {
            "year": year,
            "window_id": f"{zone}_{shard_col}_{shard_row}",
            "zone": zone,
            "col": shard_col,
            "row": shard_row,
        }
        # Pixel rows grow southward, so northing is the negated pixel y.
        x_res, y_res = projection.x_resolution, projection.y_resolution
        windows.append(
            {
                **key,
                "crs": crs,
                "block": block,
                "utm_xmin": round(core[0] * x_res),
                "utm_ymin": round(core[3] * y_res),
                "utm_xmax": round(core[2] * x_res),
                "utm_ymax": round(core[1] * y_res),
                "context_pixels": context_pixels,
                "written": (core[0], core[1]) in written,
                "inference_crop_size": marker.get("window_size"),
                "geometry": shapely.to_wkb(footprint),
                "bbox": _bbox(footprint),
            }
        )
        for modality, groups in layers.items():
            for group in groups:
                period = group["time_range"] or [None, None]
                for mosaic_order, item in enumerate(group["items"]):
                    scene_id = item["name"]
                    window_scenes.append(
                        {
                            **key,
                            "modality": modality,
                            "period_start": _date(period[0]),
                            "period_end": _date(period[1]),
                            "mosaic_order": mosaic_order,
                            "scene_id": scene_id,
                        }
                    )
                    if scene_id not in scenes:
                        geometry = STGeometry.deserialize(item["geometry"])
                        shp = _wgs84(geometry)
                        scenes[scene_id] = {
                            "year": year,
                            "scene_id": scene_id,
                            "modality": modality,
                            "collected_at": (
                                geometry.time_range[0]
                                if geometry.time_range is not None
                                else None
                            ),
                            "geometry": shapely.to_wkb(shp),
                            "bbox": _bbox(shp),
                        }
    return windows, window_scenes, scenes


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


def _stac_item_url(data_provider: str, collection: str, vendor_id: str) -> str | None:
    """The scene's item in a public STAC catalog, if its provider has one we know."""
    if data_provider == "planetary_computer":
        return PLANETARY_COMPUTER_ITEM_URL.format(
            collection=collection, vendor_id=vendor_id
        )
    template = STAC_ITEM_URLS.get((data_provider, collection))
    return template.format(vendor_id=vendor_id) if template else None


def _resolve(scene: dict, record: dict | None, excludes: set[str]) -> dict:
    """Add a scene's public source from its datasets service record.

    A scene the service no longer has, or one whose only providers are excluded, keeps
    null source columns.

    Raises:
        ValueError: if the config allows more than one provider for the scene. The
            reader then picked one at random per read, so the source is unknown.
    """
    empty = {
        "satellite": None,
        "product_id": None,
        "cloud_cover": None,
        "data_provider": None,
        "collection": None,
        "vendor_id": None,
        "stac_item_url": None,
    }
    if record is None:
        return {**scene, **empty}
    allowed = sorted(
        (name, provider)
        for name, provider in record["data_providers"].items()
        if name not in excludes
    )
    if len(allowed) > 1:
        raise ValueError(
            f"{scene['scene_id']} has {len(allowed)} allowed providers "
            f"({', '.join(name for name, _ in allowed)}), so its source is unknown"
        )
    properties = record.get("properties", {})
    known = {
        "satellite": properties.get("satellite"),
        # Only Sentinel-2 records carry one, the ESA SAFE name; elsewhere vendor_id
        # is already the product name.
        "product_id": properties.get("product_id"),
        "cloud_cover": properties.get("cloud_cover"),
    }
    if not allowed:
        return {**scene, **empty, **known}
    ((name, provider),) = allowed
    collection = provider.get("collection")
    return {
        **scene,
        **known,
        "data_provider": name,
        "collection": collection,
        "vendor_id": provider["id"],
        "stac_item_url": _stac_item_url(name, collection, provider["id"]),
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


def _write_geo_table(
    table: pa.Table, filesystem: Any, path: str, row_group_size: int
) -> int:
    table = table.replace_schema_metadata({b"geo": _geo_metadata(table)})
    filesystem.create_dir(path.rsplit("/", 1)[0], recursive=True)
    pq.write_table(
        table,
        path,
        filesystem=filesystem,
        compression=COMPRESSION,
        row_group_size=row_group_size,
    )
    return table.num_rows


def _zone_of_block(name: str) -> int:
    """The EPSG code of a block name such as "EPSG:32631_32768_-540672.json"."""
    return int(name.split("_", 1)[0].removeprefix("EPSG:"))


def _batches(values: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def export_provenance(
    run_path: str,
    year: int,
    out_dir: str,
    workers: int = 64,
    inputs: EmbeddingInputs = EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
    resolve_sources: bool = True,
) -> None:
    """Export one year's provenance to windows, window_scenes and scenes GeoParquet tables.

    Resolving sources reads the datasets service's URL and token from the same
    environment variables the workers use (OEDATASETS_API_URL, DATASETS_API_TOKEN).

    Args:
        run_path: the directory holding completed_<year>/ and provenance_<year>/.
        year: the reference year to export.
        out_dir: where to write the tables, a local path or a pyarrow filesystem URI.
        workers: threads reading marker and provenance files.
        inputs: the input variant the run used, whose dataset config names the
            providers each modality excluded.
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
        return _block_rows(marker, provenance, name.removesuffix(".json"), year)

    filesystem, root = pyarrow.fs.FileSystem.from_uri(out_dir)
    fname = f"{year}.parquet"
    window_scenes_path = f"{root}/window_scenes/{fname}"
    filesystem.create_dir(window_scenes_path.rsplit("/", 1)[0], recursive=True)

    # One zone at a time, which bounds memory and keeps row groups within a zone.
    by_zone: dict[int, list[str]] = {}
    for name in blocks:
        by_zone.setdefault(_zone_of_block(name), []).append(name)

    windows: list[dict] = []
    scenes: dict[str, dict] = {}
    n_window_scenes = 0
    n_read = 0
    with (
        ThreadPoolExecutor(workers) as pool,
        pq.ParquetWriter(
            window_scenes_path,
            WINDOW_SCENES_SCHEMA,
            filesystem=filesystem,
            compression=COMPRESSION,
        ) as window_scenes_writer,
    ):
        for zone in sorted(by_zone):
            window_scene_rows: list[dict] = []
            for block_windows, block_window_scenes, block_scenes in pool.map(
                read, by_zone[zone]
            ):
                windows += block_windows
                window_scene_rows += block_window_scenes
                for scene_id, scene in block_scenes.items():
                    scenes.setdefault(scene_id, scene)
            if window_scene_rows:
                window_scenes_writer.write_table(
                    pa.Table.from_pylist(
                        window_scene_rows, schema=WINDOW_SCENES_SCHEMA
                    ).sort_by(WINDOW_SCENES_SORT),
                    row_group_size=WINDOW_SCENES_ROW_GROUP,
                )
            n_window_scenes += len(window_scene_rows)
            n_read += len(by_zone[zone])
            logger.info(
                "%d of %d blocks read, %d window_scenes rows, %d scenes",
                n_read,
                len(blocks),
                n_window_scenes,
                len(scenes),
            )

    windows_table = pa.Table.from_pylist(windows, schema=WINDOWS_SCHEMA)
    n_windows = _write_geo_table(
        windows_table.take(_hilbert_order(windows_table)),
        filesystem,
        f"{root}/windows/{fname}",
        WINDOWS_ROW_GROUP,
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
        _resolve(scene, records.get(scene_id), excludes.get(scene["modality"], set()))
        for scene_id, scene in sorted(scenes.items())
    ]
    n_scenes = _write_geo_table(
        pa.Table.from_pylist(resolved, schema=SCENES_SCHEMA),
        filesystem,
        f"{root}/scenes/{fname}",
        SCENES_ROW_GROUP,
    )
    logger.info(
        "wrote %d windows, %d window_scenes rows and %d scenes for %d to %s",
        n_windows,
        n_window_scenes,
        n_scenes,
        year,
        out_dir,
    )
