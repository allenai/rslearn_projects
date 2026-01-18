"""
Create windows for landslide detection (segmentation task) - one window per sample, window size max(64, landslide polygon extent).

Data source is Sen12Landslides dataset.

For each landslide event, there are 2 window types:
1. Negative window: 1 year before event, 60 day window (no_landslide label for entire window)
2. Positive window: After event, 60 day window (landslide polygon labeled, rest is nodata)

python create_windows_for_landslide_detection.py \
    --shapefile_path /weka/dfive-default/piperw/data/landslide/sen12landslides/inventories.shp \
    --ds_path data/landslide/20260113_one_per_sample/ \
    --buffer_factor 1.5 \
    --sample_type positive \
    --max_samples 100
"""

import argparse
import multiprocessing
from datetime import datetime, timedelta, timezone

import geopandas as gpd
import pandas as pd
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.utils.windows import calculate_bounds

WINDOW_RESOLUTION = 10
MIN_WINDOW_SIZE_PIXELS = 64
LABEL_LAYER = "label"


def create_window_pair(
    row_data: dict, 
    dataset: Dataset, 
    buffer_factor: float, 
    sample_type: str,
) -> None:
    """Create pre-event and post-event windows for landslide detection.

    Args:
        row_data: dictionary containing landslide event data
        dataset: the dataset to create the windows in
        buffer_factor: multiplier for polygon extent (e.g., 1.5 = 50% buffer)
        sample_type: "positive" or "negative", which windows to create
    """
    sample_id = row_data["id"]
    latitude, longitude = row_data["latitude"], row_data["longitude"]
    event_date = row_data["event_date"]
    event_type = row_data["event_type"]
    location = row_data["location"]
    geometry = row_data["geometry"]
    
    sampling_date = pd.to_datetime(event_date).to_pydatetime().replace(tzinfo=timezone.utc)
    event_year = sampling_date.year
    
    # Create spatial geometry centered on the landslide centroid
    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)

    # Calculate window size - use at least MIN_WINDOW_SIZE_PIXELS
    window_size = MIN_WINDOW_SIZE_PIXELS  # At least 64 pixels on a side
    max_extent = window_size * WINDOW_RESOLUTION  # for logging, in meters
    bounds = calculate_bounds(dst_geometry, window_size)
    
    # Verify the polygon fits within the window (should be true with buffer_factor and min size)
    print(f"  Window size: {window_size} pixels (~{max_extent:.2f} m extent)")

    is_val = event_year >= 2021
    split = "val" if is_val else "train"

    group = "sen12_landslides"
    
    if sample_type == "negative":
        # Negative window: 1 year before event, 60 day window
        negative_start_time = sampling_date.replace(year=sampling_date.year - 1)
        negative_end_time = negative_start_time + timedelta(days=60)
        
        negative_window_name = f"{sample_id}_negative_{latitude:.4f}_{longitude:.4f}_{event_year}"
        
        print(f"Creating NEGATIVE window: {negative_window_name}")
        print(f"  Time range: {negative_start_time} to {negative_end_time} (1 year before event, 60 days)")
        print(f"  Window size: {window_size} pixels ({max_extent:.2f}m extent)")
        
        negative_window = Window(
            storage=dataset.storage,
            group=group,
            name=negative_window_name,
            projection=dst_projection,
            bounds=bounds,
            time_range=(negative_start_time, negative_end_time),
            options={
                "split": split,
                "latitude": latitude,
                "longitude": longitude,
                "event_date": event_date.isoformat() if hasattr(event_date, 'isoformat') else str(event_date),
                "event_type": event_type,
                "location": location,
                "event_year": event_year,
                "window_size": window_size,
                "polygon_extent_m": max_extent,
                "window_type": "negative",
                "time_range_start": negative_start_time.isoformat(),
                "time_range_end": negative_end_time.isoformat(),
            },
        )
        negative_window.save()

        # Negative: entire window is labeled as "no_landslide"
        negative_feature = Feature(
            negative_window.get_geometry(),
            {
                "label": "no_landslide",
                "event_type": event_type,
                "event_date": str(event_date),
            },
        )
        negative_layer_dir = negative_window.get_layer_dir(LABEL_LAYER)
        GeojsonVectorFormat().encode_vector(negative_layer_dir, [negative_feature])
        negative_window.mark_layer_completed(LABEL_LAYER)

    elif sample_type == "positive":
        # Positive window: Event date + 60 days after
        positive_start_time = sampling_date
        positive_end_time = sampling_date + timedelta(days=60)
        
        positive_window_name = f"{sample_id}_positive_{latitude:.4f}_{longitude:.4f}_{event_year}"
        
        print(f"Creating POSITIVE window: {positive_window_name}")
        print(f"  Time range: {positive_start_time} to {positive_end_time} (event date + 60 days)")
        print(f"  (pre_sentinel2 will query 1 year before this via config time_offset)")
        print(f"  (post_sentinel2 will query during this window)")
        print(f"  Window size: {window_size} pixels ({max_extent:.2f}m extent)")
        
        positive_window = Window(
            storage=dataset.storage,
            group=group,
            name=positive_window_name,
            projection=dst_projection,
            bounds=bounds,
            time_range=(positive_start_time, positive_end_time),
            options={
                "split": split,
                "latitude": latitude,
                "longitude": longitude,
                "event_date": event_date.isoformat() if hasattr(event_date, 'isoformat') else str(event_date),
                "event_type": event_type,
                "location": location,
                "event_year": event_year,
                "window_size": window_size,
                "polygon_extent_m": max_extent,
                "window_type": "positive",
                "time_range_start": positive_start_time.isoformat(),
                "time_range_end": positive_end_time.isoformat(),
            },
        )
        positive_window.save()

        # Positive: only the landslide polygon is labeled as "landslide"
        # Everything else in the window is nodata
        label_geometry = STGeometry(WGS84_PROJECTION, geometry, None)
        positive_feature = Feature(
            label_geometry,
            {
                "label": "landslide",
                "event_type": event_type,
                "event_date": str(event_date),
            },
        )
        positive_layer_dir = positive_window.get_layer_dir(LABEL_LAYER)
        GeojsonVectorFormat().encode_vector(positive_layer_dir, [positive_feature])
        positive_window.mark_layer_completed(LABEL_LAYER)

    else:
        print("WARNING: sample_type must be 'positive' or 'negative'")
    
    print(f"✓ Created window for sample {sample_id}\n")


def create_windows_from_shapefile(
    shapefile_path: UPath,
    ds_path: UPath,
    buffer_factor: float,
    sample_type: str,
    max_samples: int = None,
) -> None:
    """Create windows from Sen12Landslides shapefile.

    Args:
        shapefile_path: path to the inventories.shp file
        ds_path: path to the dataset
        buffer_factor: multiplier for polygon extent to add context
        sample_type: "positive" or "negative", which windows to create
        max_samples: maximum number of samples to process (None for all)
    """
    gdf = gpd.read_file(shapefile_path)
    
    gdf["centroid"] = gdf.geometry.centroid
    gdf["latitude"] = gdf["centroid"].y
    gdf["longitude"] = gdf["centroid"].x
    gdf["event_date"] = pd.to_datetime(gdf["event_date"], errors="coerce")
    
    gdf = gdf.dropna(subset=["event_date"])
    
    print(f"Total landslide events: {len(gdf)}")
    
    if max_samples is not None and max_samples < len(gdf):
        gdf = gdf.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} events")
    
    # Convert to list of dictionaries for processing
    rows_data = []
    for _, row in gdf.iterrows():
        rows_data.append({
            "id": row["id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "event_date": row["event_date"],
            "event_type": row.get("event_type", "unknown"),
            "location": row.get("location", "unknown"),
            "geometry": row["geometry"],
        })

    dataset = Dataset(ds_path)
    jobs = [
        dict(
            row_data=row,
            dataset=dataset,
            buffer_factor=buffer_factor,
            sample_type=sample_type,
        )
        for row in rows_data
    ]
    
    p = multiprocessing.Pool(32)
    outputs = star_imap_unordered(p, create_window_pair, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(description="Create windows for landslide detection")
    parser.add_argument(
        "--shapefile_path",
        type=str,
        required=True,
        help="Path to the inventories.shp file from Sen12Landslides",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--buffer_factor",
        type=float,
        required=False,
        default=1.5,
        help="Multiplier for polygon extent to add context (default: 1.5 = 50%% buffer)",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        required=True,
        help="positive or negative, which windows to create",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        required=False,
        default=None,
        help="Maximum number of samples to process (default: None = all)",
    )
    args = parser.parse_args()
    
    create_windows_from_shapefile(
        UPath(args.shapefile_path),
        UPath(args.ds_path),
        buffer_factor=args.buffer_factor,
        sample_type=args.sample_type,
        max_samples=args.max_samples,
    )
