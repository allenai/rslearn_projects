"""
Create windows for landslide detection (segmentation task).

For each landslide event, creates two windows:
1. Negative window: 1 year before event date -> + 60 days (no landslides expected)
2. Positive window: event date -> + 60 days after event (landslides expected)

Both windows check for overlapping landslides within 60 days before the current event date.
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
DEFAULT_BUFFER_DISTANCE = 30.0  # meters (3 pixels at 10m/pixel resolution)


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
    
    def query_overlapping(
        self, 
        window_geometry: shapely.Geometry, 
        time_range: tuple[datetime, datetime] = None
    ) -> List[Dict]:
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
                    "id": str(row["id"]),
                    "geometry": row["geometry"],
                    "event_type": str(row.get("event_type", "unknown")),
                    "event_date": row.get("event_date"),
                })
        
        return overlapping


def create_windows_for_landslide(
    row_data: dict,
    dataset: Dataset,
    spatial_index: LandslideSpatialIndex,
    buffer_distance: float = DEFAULT_BUFFER_DISTANCE,
) -> None:
    """Create negative and positive windows for a single landslide event.

    Args:
        row_data: dictionary containing landslide event data
        dataset: the dataset to create the windows in
        spatial_index: spatial index of all landslide polygons
        buffer_distance: distance in meters to buffer around landslides for no_data zone
    """
    sample_id = str(row_data["id"])
    latitude = float(row_data["latitude"])
    longitude = float(row_data["longitude"])
    event_date = row_data["event_date"]
    event_type = str(row_data["event_type"])
    location = str(row_data["location"])
    geometry = row_data["geometry"]
    
    # Parse event date
    sampling_date = pd.to_datetime(event_date).to_pydatetime().replace(tzinfo=timezone.utc)
    event_year = int(sampling_date.year)
    
    # Determine train/val split
    is_val = event_year >= 2021
    split = "val" if is_val else "train"
    
    # Create projection for this location
    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    
    # Calculate window bounds
    bounds = calculate_bounds(dst_geometry, WINDOW_SIZE_PIXELS)
    max_extent = WINDOW_SIZE_PIXELS * WINDOW_RESOLUTION  # in meters
    
    # Convert bounds to WGS84 for spatial queries
    if hasattr(bounds, 'min_x'):
        min_x, min_y, max_x, max_y = bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y
    else:
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
    window_geom_wgs84 = shapely.ops.transform(transformer.transform, window_geom_projected)
    
    # Define time ranges for overlapping landslides (60 days before event date)
    overlap_start_time = sampling_date - timedelta(days=60)
    overlap_end_time = sampling_date
    
    # Query for overlapping landslides (within 60 days before event date)
    overlapping_landslides = spatial_index.query_overlapping(
        window_geom_wgs84,
        time_range=(overlap_start_time, overlap_end_time)
    )
    
    # Ensure primary landslide is always included
    primary_landslide = {
        "id": sample_id,
        "geometry": geometry,
        "event_type": event_type,
        "event_date": event_date,
    }
    # Remove primary if already in list, then add at beginning
    overlapping_landslides = [ls for ls in overlapping_landslides if ls["id"] != sample_id]
    overlapping_landslides.insert(0, primary_landslide)
    
    group = "sen12_landslides"
    
    # ===== CREATE NEGATIVE WINDOW =====
    negative_start_time = sampling_date.replace(year=sampling_date.year - 1)
    negative_end_time = negative_start_time + timedelta(days=60)
    
    negative_window_name = f"{sample_id}_negative_{latitude:.4f}_{longitude:.4f}_{event_year}"
    print(f"Creating NEGATIVE window: {negative_window_name}")
    print(f"  Time range: {negative_start_time} to {negative_end_time}")
    print(f"  Found {len(overlapping_landslides)} overlapping landslides (within 60 days before event)")
    
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
            "window_size": int(WINDOW_SIZE_PIXELS),
            "polygon_extent_m": float(max_extent),
            "window_type": "negative",
            "time_range_start": negative_start_time.isoformat(),
            "time_range_end": negative_end_time.isoformat(),
            "num_overlapping_landslides": int(len(overlapping_landslides)),
            "buffer_distance_m": float(buffer_distance),
        },
    )
    negative_window.save()
    
    # Create label features for negative window
    negative_features = create_label_features(
        overlapping_landslides,
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
    
    # ===== CREATE POSITIVE WINDOW =====
    positive_start_time = sampling_date
    positive_end_time = sampling_date + timedelta(days=60)
    
    positive_window_name = f"{sample_id}_positive_{latitude:.4f}_{longitude:.4f}_{event_year}"
    print(f"Creating POSITIVE window: {positive_window_name}")
    print(f"  Time range: {positive_start_time} to {positive_end_time}")
    print(f"  Found {len(overlapping_landslides)} overlapping landslides (within 60 days before event)")
    
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
            "window_size": int(WINDOW_SIZE_PIXELS),
            "polygon_extent_m": float(max_extent),
            "window_type": "positive",
            "time_range_start": positive_start_time.isoformat(),
            "time_range_end": positive_end_time.isoformat(),
            "num_overlapping_landslides": int(len(overlapping_landslides)),
            "buffer_distance_m": float(buffer_distance),
        },
    )
    positive_window.save()
    
    # Create label features for positive window
    positive_features = create_label_features(
        overlapping_landslides,
        positive_window,
        buffer_distance,
        dst_crs,
        sample_id,
        event_type,
        event_date
    )
    
    # Verify positive window has at least one landslide
    landslide_features = [f for f in positive_features if f.properties.get("label") == "landslide"]
    if len(landslide_features) == 0:
        raise ValueError(
            f"Positive window {positive_window_name} has no landslide features! "
            f"Primary landslide {sample_id} should always be included."
        )
    
    positive_layer_dir = positive_window.get_layer_dir(LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(positive_layer_dir, positive_features)
    positive_window.mark_layer_completed(LABEL_LAYER)
    
    print(f"✓ Created windows for sample {sample_id}\n")


def create_label_features(
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
        overlapping_landslides: List of landslide dictionaries (must include primary)
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
    
    num_landslides = sum(1 for f in features if f.properties['label'] == 'landslide')
    num_buffers = sum(1 for f in features if f.properties['label'] == 'no_data')
    num_background = sum(1 for f in features if f.properties['label'] == 'no_landslide')
    print(f"    Created {len(features)} label features: {num_landslides} landslides, "
          f"{num_buffers} buffer, {num_background} background")
    
    return features


def create_windows_from_shapefile(
    shapefile_path: UPath,
    ds_path: UPath,
    max_samples: int = None,
    buffer_distance: float = DEFAULT_BUFFER_DISTANCE,
) -> None:
    """Create windows from Sen12Landslides shapefile.

    Args:
        shapefile_path: path to the inventories.shp file
        ds_path: path to the dataset
        max_samples: maximum number of samples to process (None for all)
        buffer_distance: distance in meters to buffer around landslides
    """
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Calculate centroids and extract coordinates
    gdf["centroid"] = gdf.geometry.centroid
    gdf["latitude"] = gdf["centroid"].y
    gdf["longitude"] = gdf["centroid"].x
    gdf["event_date"] = pd.to_datetime(gdf["event_date"], errors="coerce")
    
    # Drop rows with missing event dates
    gdf = gdf.dropna(subset=["event_date"])
    
    print(f"Total landslide events: {len(gdf)}")
    
    # Build spatial index from all landslides
    spatial_index = LandslideSpatialIndex(gdf)
    
    # Sample if requested
    if max_samples is not None and max_samples < len(gdf):
        gdf = gdf.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} events for window creation")
    
    # Convert to list of dictionaries for processing
    rows_data = []
    for _, row in gdf.iterrows():
        rows_data.append({
            "id": str(row["id"]),
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
            spatial_index=spatial_index,
            buffer_distance=buffer_distance,
        )
        for row in rows_data
    ]
    
    # Process in parallel
    p = multiprocessing.Pool(32)
    outputs = star_imap_unordered(p, create_windows_for_landslide, jobs)
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
        max_samples=args.max_samples,
        buffer_distance=args.buffer_distance,
    )

