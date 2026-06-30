"""Upload change finder windows to OlmoEarth Studio.

For each window, creates a Studio task and uploads per-timestep true-color RGB
composites for configurable years, plus the change-mask GeoTIFF produced by
compute_embeddings.py.

Safe to re-run: existing tasks with matching names are deleted before re-creation.
"""

import argparse
import io
import json
import multiprocessing
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import affine
import numpy as np
import rasterio
import requests
import tqdm
from rslearn.dataset import Dataset
from rslearn.utils.raster_format import get_bandset_dirname
from upath import UPath

BAND_ORDER = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
BANDSET_DIR = get_bandset_dirname(BAND_ORDER)
# B04=red (index 3), B03=green (index 2), B02=blue (index 1)
RGB_BAND_INDICES = [3, 2, 1]
NUM_TIMESTEPS = 6
RGB_CLIP_MAX = 3000
TIME_OFFSETS_DAYS = [0, 365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285]
LAYER_DURATION_DAYS = 180
PERIOD_DAYS = 30
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


def _get_headers() -> dict[str, str]:
    api_key = os.environ["STUDIO_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def _api_request(
    method: str,
    url: str,
    retries: int = MAX_RETRIES,
    **kwargs: Any,
) -> requests.Response:
    """HTTP request with retries and exponential backoff."""
    kwargs.setdefault("headers", _get_headers())
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    for attempt in range(retries):
        resp = requests.request(method, url, **kwargs)
        if resp.status_code < 500:
            return resp
        if attempt < retries - 1:
            time.sleep(RETRY_BACKOFF * (2**attempt))
    return resp


def get_existing_tasks(api_url: str, project_id: str) -> dict[str, str]:
    """Fetch all tasks in the project. Returns {task_name: task_id}."""
    result: dict[str, str] = {}
    offset = 0
    while True:
        resp = _api_request(
            "POST",
            f"{api_url}/tasks/search",
            json={"project_id": {"eq": project_id}, "offset": offset, "limit": 1000},
        )
        resp.raise_for_status()
        records = resp.json()["records"]
        if not records:
            break
        for task in records:
            result[task["name"]] = task["id"]
        offset += len(records)
    return result


def _delete_task(api_url: str, task_id: str) -> None:
    resp = _api_request("DELETE", f"{api_url}/tasks/{task_id}")
    resp.raise_for_status()


def _create_task(
    api_url: str,
    project_id: str,
    name: str,
    geom_wkt: str,
    start_time: datetime,
    end_time: datetime,
    group: str,
) -> str:
    """Create a Studio task and return its ID."""
    resp = _api_request(
        "POST",
        f"{api_url}/tasks",
        json={
            "name": name,
            "project_id": project_id,
            "geom": geom_wkt,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "attributes": {"group": group, "window": name},
        },
    )
    resp.raise_for_status()
    return resp.json()["records"][0]["id"]


def _uint16_to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a (C, H, W) uint16 array to (3, H, W) uint8 true-color RGB."""
    rgb = array[RGB_BAND_INDICES].astype(np.float32)
    rgb = np.clip(rgb, 0, RGB_CLIP_MAX) / RGB_CLIP_MAX * 255
    return rgb.astype(np.uint8)


def _write_rgb_geotiff(rgb: np.ndarray, crs: Any, transform: affine.Affine) -> bytes:
    """Write a 3-band uint8 GeoTIFF to bytes."""
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        compress="deflate",
        width=rgb.shape[2],
        height=rgb.shape[1],
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(rgb)
    return buf.getvalue()


def _upload_image(
    api_url: str,
    task_id: str,
    tiff_bytes: bytes,
    start_time: datetime,
    end_time: datetime,
    source: str,
    attributes: dict[str, Any],
) -> None:
    """Upload a GeoTIFF to Studio as an image attached to the given task."""
    resp = _api_request(
        "POST",
        f"{api_url}/images/upload",
        data={
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "source": source,
            "attributes": json.dumps(attributes),
            "task_id": str(task_id),
        },
        files={"image_file": ("image.tif", io.BytesIO(tiff_bytes), "image/tiff")},
        timeout=120,
    )
    resp.raise_for_status()


def _read_layer_geotiff(
    window_root: UPath, layer_name: str
) -> tuple[np.ndarray, Any, affine.Affine, list[tuple[datetime, datetime]] | None]:
    """Read a multi-timestep layer GeoTIFF and return (C, T, H, W) array + CRS + transform + timestamps."""
    raster_dir = window_root / "layers" / layer_name / BANDSET_DIR
    tiff_path = raster_dir / "geotiff.tif"

    metadata_path = raster_dir / "metadata.json"
    timestamps = None
    num_channels = len(BAND_ORDER)
    num_timesteps = NUM_TIMESTEPS

    if metadata_path.exists():
        with metadata_path.open() as f:
            meta = json.load(f)
        if meta.get("num_channels") is not None:
            num_channels = meta["num_channels"]
        if meta.get("num_timesteps") is not None:
            num_timesteps = meta["num_timesteps"]
        if meta.get("timestamps") is not None:
            timestamps = [
                (datetime.fromisoformat(t[0]), datetime.fromisoformat(t[1]))
                for t in meta["timestamps"]
            ]

    # Read via a local copy if the path is remote (GCS etc.)
    if tiff_path.path.startswith("gs://") or tiff_path.path.startswith("s3://"):
        with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
            with tiff_path.open("rb") as src:
                tmp.write(src.read())
            tmp.flush()
            with rasterio.open(tmp.name) as src:
                array = src.read()
                crs = src.crs
                transform = src.transform
    else:
        with rasterio.open(str(tiff_path)) as src:
            array = src.read()
            crs = src.crs
            transform = src.transform

    # Reshape (C*T, H, W) -> (C, T, H, W)
    array = array.reshape(num_channels, num_timesteps, array.shape[1], array.shape[2])
    return array, crs, transform, timestamps


def _process_window(args: tuple) -> str:
    """Process a single window: create task, upload imagery and mask.

    Returns the window name on success.
    """
    (
        ds_path,
        api_url,
        project_id,
        group,
        window_name,
        years,
        mask_filename,
        existing_task_id,
        dry_run,
    ) = args

    dataset = Dataset(UPath(ds_path))
    windows = dataset.load_windows(groups=[group], names=[window_name])
    if not windows:
        print(f"  WARNING: window {group}/{window_name} not found, skipping")
        return window_name

    window = windows[0]
    window_root = dataset.storage.get_window_root(group, window_name)

    # Compute WGS84 geometry
    geom_wkt = window.get_geometry().to_wgs84().shp.wkt

    # Task time range: from base_time to end of the latest requested year
    base_time = window.time_range[0]
    max_year = max(years)
    task_end = base_time + timedelta(
        days=TIME_OFFSETS_DAYS[max_year] + LAYER_DURATION_DAYS
    )

    if dry_run:
        print(f"  [DRY RUN] Would create task '{window_name}' ({geom_wkt[:60]}...)")
        print(
            f"            time_range: {base_time.isoformat()} - {task_end.isoformat()}"
        )
        for yi in years:
            print(
                f"            Would upload {NUM_TIMESTEPS} RGB images for sentinel2_y{yi}"
            )
        if mask_filename:
            mask_path = window_root / mask_filename
            print(
                f"            Would upload mask: {mask_path} (exists={mask_path.exists()})"
            )
        return window_name

    # Delete existing task if present
    if existing_task_id:
        _delete_task(api_url, existing_task_id)

    # Create task
    task_id = _create_task(
        api_url=api_url,
        project_id=project_id,
        name=window_name,
        geom_wkt=geom_wkt,
        start_time=base_time,
        end_time=task_end,
        group=group,
    )

    # Upload imagery for each requested year
    for yi in years:
        layer_name = f"sentinel2_y{yi}"
        try:
            array, crs, transform, timestamps = _read_layer_geotiff(
                window_root, layer_name
            )
        except Exception as e:
            print(f"  WARNING: could not read {layer_name} for {window_name}: {e}")
            continue

        year_offset = timedelta(days=TIME_OFFSETS_DAYS[yi])
        actual_num_timesteps = array.shape[1]

        for t in range(actual_num_timesteps):
            rgb = _uint16_to_rgb_uint8(array[:, t, :, :])
            tiff_bytes = _write_rgb_geotiff(rgb, crs, transform)

            if timestamps and t < len(timestamps):
                ts_start, ts_end = timestamps[t]
            else:
                ts_start = base_time + year_offset + timedelta(days=PERIOD_DAYS * t)
                ts_end = ts_start + timedelta(days=PERIOD_DAYS)

            # Ensure timezone-aware
            if ts_start.tzinfo is None:
                ts_start = ts_start.replace(tzinfo=timezone.utc)
            if ts_end.tzinfo is None:
                ts_end = ts_end.replace(tzinfo=timezone.utc)

            _upload_image(
                api_url=api_url,
                task_id=task_id,
                tiff_bytes=tiff_bytes,
                start_time=ts_start,
                end_time=ts_end,
                source=layer_name,
                attributes={"year_index": yi, "timestep": t},
            )

    # Upload change mask
    if mask_filename:
        mask_path = window_root / mask_filename
        if mask_path.exists():
            with mask_path.open("rb") as f:
                mask_bytes = f.read()
            _upload_image(
                api_url=api_url,
                task_id=task_id,
                tiff_bytes=mask_bytes,
                start_time=base_time,
                end_time=task_end,
                source="change_mask",
                attributes={},
            )
        else:
            print(f"  WARNING: mask file {mask_path} not found for {window_name}")

    return window_name


def upload_to_studio(
    ds_path: str,
    windows_json: str,
    project_id: str,
    years: list[int],
    mask_filename: str | None = "embeddings.tif",
    workers: int = 16,
    api_url: str = "https://olmoearth.allenai.org/api/v1",
    dry_run: bool = False,
) -> None:
    """Upload change finder windows to OlmoEarth Studio."""
    with open(windows_json) as f:
        window_specs = json.load(f)
    print(f"Loaded {len(window_specs)} windows from {windows_json}")

    # Fetch existing tasks for idempotency
    if not dry_run:
        print("Fetching existing tasks for idempotent re-run...")
        existing_tasks = get_existing_tasks(api_url, project_id)
        print(f"Found {len(existing_tasks)} existing tasks in project")
    else:
        existing_tasks = {}

    jobs = []
    for spec in window_specs:
        group = spec["group"]
        name = spec["name"]
        jobs.append(
            (
                ds_path,
                api_url,
                project_id,
                group,
                name,
                years,
                mask_filename,
                existing_tasks.get(name),
                dry_run,
            )
        )

    if workers <= 1:
        for job in tqdm.tqdm(jobs, desc="Uploading windows"):
            _process_window(job)
    else:
        with multiprocessing.Pool(workers) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(_process_window, jobs),
                total=len(jobs),
                desc="Uploading windows",
            ):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload change finder windows to OlmoEarth Studio"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--windows_json", required=True, help="JSON file with window specs"
    )
    parser.add_argument("--project_id", required=True, help="Studio project UUID")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[0, 1, 8, 9],
        help="Year indices to upload imagery for (default: 0 1 8 9)",
    )
    parser.add_argument(
        "--mask_filename",
        default="embeddings.tif",
        help="Change mask filename per window",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of parallel workers"
    )
    parser.add_argument(
        "--api_url",
        default="https://olmoearth.allenai.org/api/v1",
        help="Studio API base URL",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without making API calls",
    )
    args = parser.parse_args()

    upload_to_studio(
        ds_path=args.ds_path,
        windows_json=args.windows_json,
        project_id=args.project_id,
        years=args.years,
        mask_filename=args.mask_filename,
        workers=args.workers,
        api_url=args.api_url,
        dry_run=args.dry_run,
    )
