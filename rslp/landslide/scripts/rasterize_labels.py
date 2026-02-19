"""Create label_raster from label vector data for landslide segmentation.

This script processes window directories directly, reading data.geojson from each window
and rasterizing the geometries to create label_raster layers.
"""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
import shapely
import tqdm
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat, get_transform_from_projection_and_bounds
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

# Band name for the raster output
BAND_NAME = "label"

# Property name in the geojson features that contains the class label
CLASS_PROPERTY_NAME = "label"

# Class mapping: map from geojson property values to raster pixel values
# Based on config.json: ["no_data", "no_landslide", "landslide"]
# Maps to: 2 (nodata), 0 (background), 1 (landslide)
CLASS_MAPPING = {
    "no_data": 2,
    "no_landslide": 0,
    "landslide": 1,
}


def get_class_value(feature_properties: dict) -> int:
    """Get the class value from feature properties.
    
    Args:
        feature_properties: The properties dictionary from a geojson feature
        
    Returns:
        Integer class value for rasterization
    """
    label = feature_properties.get(CLASS_PROPERTY_NAME)
    
    # If label is already an integer, use it directly
    if isinstance(label, (int, np.integer)):
        return int(label)
    
    # If label is a string, try to map it
    if isinstance(label, str):
        mapped_value = CLASS_MAPPING.get(label)
        if mapped_value is not None:
            return mapped_value
        # If not found in mapping, default to background
        print(f"Warning: Unknown label value '{label}', defaulting to 0")
        return 0
    
    # Default fallback (background)
    return 0


def load_window_metadata(window_dir: UPath) -> tuple[Projection, tuple[int, int, int, int]]:
    """Load projection and bounds from window metadata.json.
    
    Args:
        window_dir: Path to the window directory
        
    Returns:
        Tuple of (projection, bounds)
    """
    metadata_path = window_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {window_dir}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Load projection
    proj_dict = metadata["projection"]
    projection = Projection.deserialize(proj_dict)
    
    # Load bounds (convert from list to tuple)
    bounds = tuple(metadata["bounds"])
    
    return projection, bounds


def create_label_raster(window_dir: UPath) -> None:
    """Create label raster for the given window directory.
    
    Args:
        window_dir: Path to the window directory
    """
    window_dir = UPath(window_dir)
    
    # Load window metadata to get projection and bounds
    try:
        projection, bounds = load_window_metadata(window_dir)
    except Exception as e:
        print(f"Error loading metadata for {window_dir}: {e}")
        return
    
    # Read vector features from data.geojson
    label_dir = window_dir / "layers" / "label"
    geojson_path = label_dir / "data.geojson"
    
    if not geojson_path.exists():
        print(f"Warning: data.geojson not found in {label_dir}, skipping {window_dir.name}")
        return
    
    # Read geojson and decode features
    features = GeojsonVectorFormat().decode_vector(
        label_dir, projection, bounds
    )
    
    if not features:
        # No features in this window - create empty raster with background class (0 = no_landslide)
        height = bounds[3] - bounds[1]
        width = bounds[2] - bounds[0]
        raster = np.zeros((1, height, width), dtype=np.uint8)
    else:
        # Prepare shapes for rasterization: (geometry, value) tuples.
        # decode_vector returns geometries in the window's projection, where .shp is in
        # *pixel* coordinates (same units as bounds). rasterio.rasterize expects
        # geometries in *CRS* coordinates for the given transform. Convert pixel -> CRS
        # by scaling: x_crs = x_pixel * x_resolution, y_crs = y_pixel * y_resolution.
        transform = get_transform_from_projection_and_bounds(projection, bounds)
        height = bounds[3] - bounds[1]
        width = bounds[2] - bounds[0]

        def pixel_to_crs(geom: shapely.Geometry) -> shapely.Geometry:
            if geom is None or geom.is_empty:
                return geom
            return shapely.affinity.scale(
                geom,
                xfact=projection.x_resolution,
                yfact=projection.y_resolution,
                origin=(0, 0),
            )

        shapes = []
        for feature in features:
            class_value = get_class_value(feature.properties)
            geom_pixel = feature.geometry.to_projection(projection)
            geom_crs = pixel_to_crs(geom_pixel.shp)
            if geom_crs is not None and not geom_crs.is_empty:
                shapes.append((geom_crs, class_value))

        # Rasterize the geometries (in CRS coordinates)
        rasterized = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,  # Background value
            all_touched=True,  # Include pixels that touch the geometry
            dtype=np.uint8,
        )
        
        # Add channel dimension: (1, H, W)
        raster = np.expand_dims(rasterized, axis=0).astype(np.uint8)
    
    # Save the raster to {window_dir}/layers/label_raster/label/geotiff.tif
    raster_dir = window_dir / "layers" / "label_raster" / BAND_NAME
    raster_dir.mkdir(parents=True, exist_ok=True)
    
    GeotiffRasterFormat().encode_raster(
        raster_dir, projection, bounds, raster
    )
    
    # Mark layer as completed
    completed_file = window_dir / "layers" / "label_raster" / "completed"
    completed_file.touch()


def process_window(window_path: Path) -> None:
    """Process a single window directory.
    
    Args:
        window_path: Path to the window directory
    """
    try:
        create_label_raster(UPath(window_path))
    except Exception as e:
        print(f"Error processing {window_path}: {e}")


def find_windows(dataset_dir: UPath) -> list[Path]:
    """Find all window directories in the dataset.
    
    Args:
        dataset_dir: Path to the dataset root directory
        
    Returns:
        List of window directory paths
    """
    windows = []
    windows_dir = dataset_dir / "windows"
    
    if not windows_dir.exists():
        raise ValueError(f"windows/ directory not found in {dataset_dir}")
    
    # Iterate through groups
    for group_dir in windows_dir.iterdir():
        if not group_dir.is_dir():
            continue
        
        # Iterate through windows in each group
        for window_dir in group_dir.iterdir():
            if window_dir.is_dir():
                windows.append(window_dir)
    
    return windows


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(
        description="Rasterize vector label data to create label_raster layer for segmentation"
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset directory (should contain windows/ subdirectory)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of worker processes to use",
    )
    parser.add_argument(
        "--class_property",
        type=str,
        default="label",
        help="Property name in geojson features that contains the class label",
    )
    args = parser.parse_args()

    # Update global class property name if provided
    CLASS_PROPERTY_NAME = args.class_property

    dataset_dir = UPath(args.ds_path)
    
    # Find all windows
    print(f"Scanning for windows in {dataset_dir}...")
    windows = find_windows(dataset_dir)
    print(f"Found {len(windows)} windows")
    
    if len(windows) == 0:
        print("No windows found. Exiting.")
        exit(1)
    
    print(f"Rasterizing labels for {len(windows)} windows...")
    print(f"Using class property: {CLASS_PROPERTY_NAME}")
    print(f"Class mapping: {CLASS_MAPPING}")
    
    # Process windows in parallel
    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(process_window, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows)):
        pass
    p.close()
    p.join()
    
    print("Done!")

