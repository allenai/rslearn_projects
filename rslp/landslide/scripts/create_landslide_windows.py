"""
Create windows for landslide detection (segmentation task) - multiple landslides in a window are all labeled as landslide.

Data source is Sen12Landslides dataset.

For each landslide event, there are 2 window types:
1. Negative window: 1 year before event, 60 day window (no_landslide label)
2. Positive window: After event, 60 day window (landslide label) - uses both pre and post sentinel2 imagery

python create_windows_for_landslide_detection.py \
    --shapefile_path /weka/dfive-default/piperw/data/landslide/sen12landslides/inventories.shp \
    --ds_path data/landslide/20260106_positives/ \
    --sample_type positive \
    --max_samples 1 \
    --buffer_distance 20  # buffer in meters around landslides (default: 20m = 2 pixels at 10m/pixel)
"""

import argparse
import multiprocessing
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import geopandas as gpd
import pandas as pd
import shapely
from shapely.strtree import STRtree
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.utils.windows import calculate_bounds

WINDOW_RESOLUTION = 10  # meters per pixel
WINDOW_SIZE_PIXELS = 64
LABEL_LAYER = "label"
DEFAULT_BUFFER_DISTANCE = 30.0  # meters (2 pixels at 10m/pixel resolution)


class LandslideSpatialIndex:
    """Spatial index for efficient lookup of overlapping landslides."""
    
    def __init__(self, gdf: gpd.GeoDataFrame):
        """Initialize spatial index from GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame containing all landslide polygons
        """
        self.gdf = gdf.copy()
        # Build spatial index using STRtree
        self.tree = STRtree(self.gdf.geometry)
        print(f"Built spatial index with {len(self.gdf)} landslide polygons")
    
    def query_overlapping(self, window_geometry: shapely.Geometry, time_range: tuple = None) -> List[Dict]:
        """Find all landslides that overlap with the given window.
        
        Args:
            window_geometry: Shapely geometry representing the window bounds
            time_range: Optional tuple of (start_time, end_time) to filter by event date
            
        Returns:
            List of dictionaries containing landslide data for overlapping polygons
        """
        # Query the spatial index
        possible_matches_idx = self.tree.query(window_geometry)
        
        # Filter to actual intersections
        overlapping = []
        for idx in possible_matches_idx:
            if self.gdf.iloc[idx].geometry.intersects(window_geometry):
                row = self.gdf.iloc[idx]
                
                # If time_range is provided, check if landslide event_date falls within it
                if time_range is not None:
                    event_date = pd.to_datetime(row.get("event_date"))
                    if pd.isna(event_date):
                        continue  # Skip if no valid event date
                    
                    start_time, end_time = time_range
                    # Convert to timezone-aware if needed
                    if event_date.tzinfo is None:
                        event_date = event_date.replace(tzinfo=timezone.utc)
                    
                    # Check if event_date is within the time range
                    if not (start_time <= event_date <= end_time):
                        continue  # Skip landslides outside the time window
                
                overlapping.append({
                    "id": str(row["id"]),  # Convert to string for JSON serialization
                    "geometry": row["geometry"],
                    "event_type": str(row.get("event_type", "unknown")),
                    "event_date": row.get("event_date"),
                })
        
        return overlapping


def create_window_pair(
    row_data: dict, 
    dataset: Dataset, 
    sample_type: str,
    spatial_index: LandslideSpatialIndex,
    buffer_distance: float = DEFAULT_BUFFER_DISTANCE,
) -> None:
    """Create pre-event and post-event windows for landslide detection.

    Args:
        row_data: dictionary containing landslide event data
        dataset: the dataset to create the windows in
        sample_type: "positive" creates both positive and negative windows, "negative" creates only negative windows
        spatial_index: spatial index of all landslide polygons
        buffer_distance: distance in meters to buffer around landslides for no_data zone
    """
    sample_id = str(row_data["id"])  # Convert to string to ensure JSON serializable
    latitude, longitude = float(row_data["latitude"]), float(row_data["longitude"])
    event_date = row_data["event_date"]
    event_type = str(row_data["event_type"])
    location = str(row_data["location"])
    geometry = row_data["geometry"]
    
    sampling_date = pd.to_datetime(event_date).to_pydatetime().replace(tzinfo=timezone.utc)
    event_year = int(sampling_date.year)
    
    # Create spatial geometry centered on the landslide centroid
    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)

    # Calculate window size based on polygon extent
    src_polygon_geometry = STGeometry(WGS84_PROJECTION, geometry, None)
    dst_polygon_geometry = src_polygon_geometry.to_projection(dst_projection)
    
    window_size = WINDOW_SIZE_PIXELS  # 64 pixels on a side
    max_extent = window_size * WINDOW_RESOLUTION  # for logging, in meters
    bounds = calculate_bounds(dst_geometry, window_size)
    print(f"  Window size: {window_size} pixels (~{max_extent:.2f} m extent)")

    is_val = event_year >= 2021
    split = "val" if is_val else "train"

    group = "sen12_landslides"
    
    # Create window geometry in projected coordinates, then transform to WGS84 for spatial query
    # bounds is typically (min_col, min_row, max_col, max_row) or an object with attributes
    if hasattr(bounds, 'min_x'):
        min_x, min_y, max_x, max_y = bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y
    else:
        # bounds is a tuple: (min_col, min_row, max_col, max_row)
        min_x, min_y, max_x, max_y = bounds[0], bounds[1], bounds[2], bounds[3]
    
    # Convert pixel coordinates to projected CRS coordinates
    proj_min_x = min_x * dst_projection.x_resolution
    proj_min_y = min_y * dst_projection.y_resolution
    proj_max_x = max_x * dst_projection.x_resolution
    proj_max_y = max_y * dst_projection.y_resolution
    
    window_geom_projected = shapely.box(proj_min_x, proj_min_y, proj_max_x, proj_max_y)
    
    # Transform back to WGS84 for querying
    from pyproj import Transformer
    transformer = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
    window_geom_wgs84_coords = shapely.ops.transform(transformer.transform, window_geom_projected)
    
    # Query for overlapping landslides with appropriate filtering
    # Always query for negative window landslides
    negative_start_time = sampling_date.replace(year=sampling_date.year - 1)
    negative_end_time = negative_start_time + timedelta(days=60)
    negative_overlapping = spatial_index.query_overlapping(
        window_geom_wgs84_coords, 
        time_range=(negative_start_time, negative_end_time)
    )
    
    # Query for positive window landslides if creating positive windows
    if sample_type == "positive":
        # For positive windows, include landslides from 180 days before up to the event date
        # This captures landslides that would be visible in the post-event imagery
        positive_start_time = sampling_date
        positive_end_time = sampling_date + timedelta(days=60)
        # Extended time range: 180 days before the event up to the event date
        extended_start_time = sampling_date - timedelta(days=180)
        positive_overlapping = spatial_index.query_overlapping(
            window_geom_wgs84_coords,
            time_range=(extended_start_time, positive_start_time)
        )
        
        # Ensure the primary landslide is always included in positive windows
        primary_landslide_found = any(ls["id"] == sample_id for ls in positive_overlapping)
        if not primary_landslide_found:
            # Add the primary landslide to the list
            primary_landslide = {
                "id": sample_id,
                "geometry": geometry,
                "event_type": event_type,
                "event_date": event_date,
            }
            positive_overlapping.append(primary_landslide)
            print(f"  Added primary landslide {sample_id} to positive window (was not in spatial query results)")
        
        print(f"  Found {len(positive_overlapping)} spatially overlapping landslides in positive window")
        print(f"  (Including landslides from 180 days before up to {event_date})")
    
    print(f"  Found {len(negative_overlapping)} overlapping landslides in negative window")
    if len(negative_overlapping) > 0:
        print(f"  WARNING: Negative window has {len(negative_overlapping)} landslides! Will label them.")
    
    # Create negative window (always created, or only when sample_type is "negative")
    if sample_type == "negative" or sample_type == "positive":
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
                "latitude": float(latitude),
                "longitude": float(longitude),
                "event_date": event_date.isoformat() if hasattr(event_date, 'isoformat') else str(event_date),
                "event_type": str(event_type),
                "location": str(location),
                "event_year": int(event_year),
                "window_size": int(window_size),
                "polygon_extent_m": float(max_extent),
                "window_type": "negative",
                "time_range_start": negative_start_time.isoformat(),
                "time_range_end": negative_end_time.isoformat(),
                "num_overlapping_landslides": int(len(negative_overlapping)),
                "buffer_distance_m": float(buffer_distance),
            },
        )
        negative_window.save()

        # Create features: landslides, buffers, and background
        negative_features = create_labeled_features(
            negative_overlapping,
            negative_window,
            buffer_distance,
            dst_crs,
            sample_id,
            event_type,
            event_date
        )
        
        negative_layer_dir = negative_window.get_layer_dir(LABEL_LAYER)
        GeojsonVectorFormat().encode_vector(negative_layer_dir, negative_features)
        negative_window.mark_layer_completed(LABEL_LAYER)

    # Create positive window (only when sample_type is "positive")
    if sample_type == "positive":
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
                "latitude": float(latitude),
                "longitude": float(longitude),
                "event_date": event_date.isoformat() if hasattr(event_date, 'isoformat') else str(event_date),
                "event_type": str(event_type),
                "location": str(location),
                "event_year": int(event_year),
                "window_size": int(window_size),
                "polygon_extent_m": float(max_extent),
                "window_type": "positive",
                "time_range_start": positive_start_time.isoformat(),
                "time_range_end": positive_end_time.isoformat(),
                "num_overlapping_landslides": int(len(positive_overlapping)),
                "buffer_distance_m": float(buffer_distance),
            },
        )
        positive_window.save()

        # Create features: landslides, buffers, and background
        positive_features = create_labeled_features(
            positive_overlapping,
            positive_window,
            buffer_distance,
            dst_crs,
            sample_id,
            event_type,
            event_date
        )
        
        # Encode all features to the label layer
        positive_layer_dir = positive_window.get_layer_dir(LABEL_LAYER)
        GeojsonVectorFormat().encode_vector(positive_layer_dir, positive_features)
        positive_window.mark_layer_completed(LABEL_LAYER)

    print(f"✓ Created window(s) for sample {sample_id}\n")


def create_labeled_features(
    overlapping_landslides: List[Dict],
    window: Window,
    buffer_distance: float,
    dst_crs: str,
    sample_id: str,
    event_type: str,
    event_date
) -> List[Feature]:
    """
    Create labeled features with three zones:
    1. Landslide polygons (label='landslide')
    2. Buffer zones around landslides (label='no_data')
    3. Background (label='no_landslide')
    
    Args:
        overlapping_landslides: List of landslide dictionaries
        window: The window object
        buffer_distance: Buffer distance in meters
        dst_crs: Destination CRS string
        sample_id: Primary sample ID
        event_type: Event type
        event_date: Event date
        
    Returns:
        List of Feature objects
    """
    from pyproj import Transformer
    
    features = []
    
    if len(overlapping_landslides) == 0:
        # No landslides - entire window is no_landslide
        features.append(Feature(
            window.get_geometry(),
            {
                "label": "no_landslide",
                "event_type": event_type,
                "event_date": str(event_date),
            },
        ))
        return features
    
    # Transform landslides to projected CRS for buffering
    transformer_to_proj = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
    
    # Collect all landslide polygons and their buffers in projected CRS
    landslide_union = None
    buffer_union = None
    
    for landslide in overlapping_landslides:
        # Get geometry in WGS84
        geom_wgs84 = landslide["geometry"]
        
        # Transform to projected CRS
        geom_proj = shapely.ops.transform(transformer_to_proj.transform, geom_wgs84)
        
        # Create landslide feature (in WGS84)
        label_geometry = STGeometry(WGS84_PROJECTION, geom_wgs84, None)
        feature = Feature(
            label_geometry,
            {
                "label": "landslide",
                "landslide_id": str(landslide["id"]),
                "event_type": str(landslide["event_type"]),
                "event_date": str(landslide["event_date"]),
                "is_primary": bool(landslide["id"] == sample_id),
            },
        )
        features.append(feature)
        
        # Union all landslides
        if landslide_union is None:
            landslide_union = geom_proj
        else:
            landslide_union = landslide_union.union(geom_proj)
        
        # Create buffer and union
        buffer_geom = geom_proj.buffer(buffer_distance)
        if buffer_union is None:
            buffer_union = buffer_geom
        else:
            buffer_union = buffer_union.union(buffer_geom)
    
    # Create buffer zone (buffer minus landslides)
    if buffer_union is not None and landslide_union is not None:
        # Buffer zone is the buffer area minus the actual landslides
        buffer_only = buffer_union.difference(landslide_union)
        
        # Transform back to WGS84
        buffer_only_wgs84 = shapely.ops.transform(transformer_to_wgs84.transform, buffer_only)
        
        # Create feature for buffer zone
        if not buffer_only_wgs84.is_empty:
            buffer_geometry = STGeometry(WGS84_PROJECTION, buffer_only_wgs84, None)
            buffer_feature = Feature(
                buffer_geometry,
                {
                    "label": "no_data",
                    "buffer_distance_m": float(buffer_distance),
                    "description": "Buffer zone around landslides",
                },
            )
            features.append(buffer_feature)
    
    # Create background (window minus landslides minus buffer)
    window_geom = window.get_geometry()
    # STGeometry has a shapely property that returns the shapely geometry
    # Or we can use the projection to get bounds and create a box
    # Let's create the window geometry directly from bounds
    window_bounds = window.bounds
    if hasattr(window_bounds, 'min_x'):
        w_min_x, w_min_y = window_bounds.min_x, window_bounds.min_y
        w_max_x, w_max_y = window_bounds.max_x, window_bounds.max_y
    else:
        w_min_x, w_min_y, w_max_x, w_max_y = window_bounds[0], window_bounds[1], window_bounds[2], window_bounds[3]
    
    # Convert to projected coordinates
    w_proj_min_x = w_min_x * window.projection.x_resolution
    w_proj_min_y = w_min_y * window.projection.y_resolution
    w_proj_max_x = w_max_x * window.projection.x_resolution
    w_proj_max_y = w_max_y * window.projection.y_resolution
    
    # Create window box in projected CRS
    window_proj = shapely.box(w_proj_min_x, w_proj_min_y, w_proj_max_x, w_proj_max_y)
    
    # Background is window minus buffer (which includes landslides)
    if buffer_union is not None:
        background = window_proj.difference(buffer_union)
    else:
        background = window_proj
    
    # Transform back to WGS84
    background_wgs84 = shapely.ops.transform(transformer_to_wgs84.transform, background)
    
    # Create feature for background
    if not background_wgs84.is_empty:
        background_geometry = STGeometry(WGS84_PROJECTION, background_wgs84, None)
        background_feature = Feature(
            background_geometry,
            {
                "label": "no_landslide",
                "event_type": event_type,
                "event_date": str(event_date),
            },
        )
        features.append(background_feature)
    
    print(f"    Created {len(features)} label features: {sum(1 for f in features if f.properties['label']=='landslide')} landslides, "
          f"{sum(1 for f in features if f.properties['label']=='no_data')} buffer, "
          f"{sum(1 for f in features if f.properties['label']=='no_landslide')} background")
    
    return features


def create_windows_from_shapefile(
    shapefile_path: UPath,
    ds_path: UPath,
    sample_type: str,
    max_samples: int = None,
    buffer_distance: float = DEFAULT_BUFFER_DISTANCE,
) -> None:
    """Create windows from Sen12Landslides shapefile.

    Args:
        shapefile_path: path to the inventories.shp file
        ds_path: path to the dataset
        sample_type: "positive" or "negative", which windows to create
        max_samples: maximum number of samples to process (None for all)
        buffer_distance: distance in meters to buffer around landslides
    """
    gdf = gpd.read_file(shapefile_path)
    
    gdf["centroid"] = gdf.geometry.centroid
    gdf["latitude"] = gdf["centroid"].y
    gdf["longitude"] = gdf["centroid"].x
    gdf["event_date"] = pd.to_datetime(gdf["event_date"], errors="coerce")
    
    gdf = gdf.dropna(subset=["event_date"])
    
    print(f"Total landslide events: {len(gdf)}")
    
    # Build spatial index from all landslides
    spatial_index = LandslideSpatialIndex(gdf)
    
    if max_samples is not None and max_samples < len(gdf):
        gdf = gdf.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} events for window creation")
    
    # Convert to list of dictionaries for processing
    rows_data = []
    for _, row in gdf.iterrows():
        rows_data.append({
            "id": str(row["id"]),  # Convert to string for JSON serialization
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "event_date": row["event_date"],
            "event_type": str(row.get("event_type", "unknown")),
            "location": str(row.get("location", "unknown")),
            "geometry": row["geometry"],
        })

    dataset = Dataset(ds_path)
    jobs = [
        dict(
            row_data=row,
            dataset=dataset,
            sample_type=sample_type,
            spatial_index=spatial_index,
            buffer_distance=buffer_distance,
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
        "--sample_type",
        type=str,
        required=True,
        help="'positive' creates both positive and negative windows, 'negative' creates only negative windows",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        required=False,
        default=None,
        help="Maximum number of samples to process (default: None = all)",
    )
    parser.add_argument(
        "--buffer_distance",
        type=float,
        required=False,
        default=DEFAULT_BUFFER_DISTANCE,
        help=f"Buffer distance in meters around landslides for no_data zone (default: {DEFAULT_BUFFER_DISTANCE}, ~{DEFAULT_BUFFER_DISTANCE/WINDOW_RESOLUTION:.1f} pixels at {WINDOW_RESOLUTION}m/pixel)",
    )
    args = parser.parse_args()
    
    create_windows_from_shapefile(
        UPath(args.shapefile_path),
        UPath(args.ds_path),
        sample_type=args.sample_type,
        max_samples=args.max_samples,
        buffer_distance=args.buffer_distance,
    )