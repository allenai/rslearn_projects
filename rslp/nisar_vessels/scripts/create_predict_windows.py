r"""Create prediction windows from NISAR scenes in a time range.

Searches the public ASF API for NISAR L2 GCOV scenes acquired in the given time
range, keeps the dual-pol H-transmit ones (the only mode with the HHHH/HVHV bands
the dataset uses), and creates one window per scene at a random location within the
scene footprint. The windows have no label layer; they are intended for the
"predict" group so the model can be applied to them (e.g. with a low confidence
threshold to mine false positives to add back to the dataset as negatives).

Window placement is deterministic per scene (seeded by the scene name), so
re-running the script does not move existing windows.

Example:
    python -m rslp.nisar_vessels.scripts.create_predict_windows \
        --ds_path /path/to/nisar_vessels_dataset/ \
        --start_time 2026-06-01T00:00:00Z \
        --end_time 2026-08-01T00:00:00Z \
        --window_size 2048 \
        --max_scenes 1000
"""

import argparse
import random
from typing import Any

import requests
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

from rslp.log_utils import get_logger

from .create_dataset import PIXEL_SIZE, TIME_BUFFER, parse_time

logger = get_logger(__name__)

ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
ASF_TIMEOUT = 300

# The polarization (POLE) code in the granule name must have this primary
# polarization for the granule to contain the HHHH/HVHV bands.
WANTED_PRIMARY_POLARIZATION = "DH"

# Number of attempts to sample a window center inside the scene footprint before
# falling back to the footprint's representative point.
MAX_PLACEMENT_ATTEMPTS = 100


def get_scenes(start_time: str, end_time: str) -> list[dict[str, Any]]:
    """Get dual-pol NISAR L2 GCOV scenes acquired in the given time range.

    Args:
        start_time: ISO-format start of the time range.
        end_time: ISO-format end of the time range.

    Returns:
        the matching ASF search results (with granuleName, wkt, startTime, and
        stopTime among other fields), de-duplicated by granule name.
    """
    response = requests.get(
        ASF_SEARCH_URL,
        params={
            "dataset": "NISAR",
            "processingLevel": "GCOV",
            "start": start_time,
            "end": end_time,
            "output": "jsonlite",
        },
        timeout=ASF_TIMEOUT,
    )
    response.raise_for_status()
    results = response.json()["results"]

    scenes: dict[str, dict[str, Any]] = {}
    for scene in results:
        name = scene["granuleName"]
        # The POLE code, e.g. DHDH, is the 10th underscore-delimited field, giving
        # the primary and secondary band polarization modes.
        pole_code = name.split("_")[9]
        if not pole_code.startswith(WANTED_PRIMARY_POLARIZATION):
            continue
        scenes[name] = scene
    logger.info(
        "got %d scenes (%d dual-pol) in %s to %s",
        len(results),
        len(scenes),
        start_time,
        end_time,
    )
    return list(scenes.values())


def sample_window_center(
    footprint: shapely.Geometry, rng: random.Random
) -> shapely.Point:
    """Sample a random point within the scene footprint.

    Args:
        footprint: the scene footprint (in WGS84).
        rng: the random number generator to use.

    Returns:
        a point inside the footprint, or its representative point if rejection
        sampling fails.
    """
    min_x, min_y, max_x, max_y = footprint.bounds
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        point = shapely.Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        if footprint.contains(point):
            return point
    return footprint.representative_point()


def create_window_for_scene(
    dataset: Dataset,
    scene: dict[str, Any],
    group: str,
    window_size: int,
) -> bool:
    """Create one randomly placed window within the scene footprint.

    Args:
        dataset: the output rslearn dataset.
        scene: the ASF search result for the scene.
        group: the group to add the window to.
        window_size: the window size in pixels.

    Returns:
        whether a window was created (False if the scene was skipped).
    """
    scene_name = scene["granuleName"]
    footprint = shapely.from_wkt(scene["wkt"])

    # Footprints crossing the antimeridian come back spanning most of the globe;
    # skip them rather than placing windows incorrectly.
    if footprint.bounds[2] - footprint.bounds[0] > 180:
        logger.warning(
            "skipping scene %s that seems to cross the antimeridian", scene_name
        )
        return False

    # Seed by scene name so re-running does not move existing windows.
    rng = random.Random(scene_name)
    center = sample_window_center(footprint, rng)

    dst_projection = get_utm_ups_projection(center.x, center.y, PIXEL_SIZE, -PIXEL_SIZE)
    center_proj = (
        STGeometry(WGS84_PROJECTION, center, None).to_projection(dst_projection).shp
    )
    bounds = (
        int(center_proj.x) - window_size // 2,
        int(center_proj.y) - window_size // 2,
        int(center_proj.x) + window_size // 2,
        int(center_proj.y) + window_size // 2,
    )
    time_range = (
        parse_time(scene["startTime"]) - TIME_BUFFER,
        parse_time(scene["stopTime"]) + TIME_BUFFER,
    )

    window = Window(
        storage=dataset.storage,
        group=group,
        name=scene_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
        options=dict(
            scene_name=scene_name,
        ),
    )
    window.save()
    return True


def create_predict_windows(
    ds_path: str,
    start_time: str,
    end_time: str,
    group: str = "predict",
    window_size: int = 2048,
    max_scenes: int | None = None,
) -> None:
    """Create one randomly placed window per NISAR scene in the time range.

    Args:
        ds_path: the path of the rslearn dataset (its config.json must already
            exist, e.g. created by create_dataset).
        start_time: ISO-format start of the time range to search for scenes.
        end_time: ISO-format end of the time range.
        group: the group to add the windows to.
        window_size: the size of each window in pixels.
        max_scenes: optionally limit to this many scenes (randomly sampled).
    """
    scenes = get_scenes(start_time, end_time)
    if max_scenes is not None and len(scenes) > max_scenes:
        # Sort first so the sample is deterministic regardless of API ordering.
        scenes.sort(key=lambda scene: scene["granuleName"])
        scenes = random.Random(0).sample(scenes, max_scenes)
        logger.info("randomly sampled %d scenes", max_scenes)

    dataset = Dataset(UPath(ds_path))
    num_created = 0
    for scene in tqdm.tqdm(scenes, desc="Creating windows"):
        if create_window_for_scene(dataset, scene, group, window_size):
            num_created += 1
    logger.info("created %d windows in group %s", num_created, group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create prediction windows from NISAR scenes in a time range",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path of the rslearn dataset (config.json must already exist)",
    )
    parser.add_argument(
        "--start_time",
        type=str,
        required=True,
        help="Start of the time range to search for scenes (ISO format)",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        required=True,
        help="End of the time range to search for scenes (ISO format)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="predict",
        help="Group to add the windows to (default: predict)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=2048,
        help="Window size in pixels (default: 2048)",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Optionally limit to this many scenes (randomly sampled)",
    )
    args = parser.parse_args()
    create_predict_windows(
        args.ds_path,
        args.start_time,
        args.end_time,
        group=args.group,
        window_size=args.window_size,
        max_scenes=args.max_scenes,
    )
