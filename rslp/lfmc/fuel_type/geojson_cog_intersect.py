"""Geospatial intersection utilities for COG rasters and vector geometries.

This module provides functions to efficiently intersect vector geometries with Cloud Optimized GeoTIFF (COG) rasters.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds
from rslearn.const import WGS84_EPSG
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union


# Configure GDAL for efficient COG access
def configure_gdal_for_cogs():
    """Configure GDAL environment variables for optimal COG performance."""
    import os
    # Enable HTTP range requests
    os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif,.tiff'
    # Cache settings for better performance
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
    os.environ['CPL_VSIL_CURL_CACHE_SIZE'] = str(256 * 2 ** 20)  # 256 MB


configure_gdal_for_cogs()


def vectorize_raster_mask(raster_path: str, threshold: int, bounds: tuple[float, float, float, float]) -> list[dict]:
    """Vectorize pixels from a raster where value > threshold.
    
    Args:
        raster_path: Path to GeoTIFF (gs://bucket/file.tif for GCS)
        threshold: Pixel value threshold (default: 0)
        bounds: Optional bounding box (left, bottom, right, top) in raster CRS 
                to limit reading. When provided, only this window is read (efficient with COGs).
        
    Returns:
        List of geometry dictionaries with CRS info
    """
    # Use /vsigs/ prefix for GCS files with rasterio
    if raster_path.startswith('gs://'):
        raster_path = '/vsigs/' + raster_path[5:]
    
    with rasterio.open(raster_path) as src:
        # Read windowed data (efficient with COGs)
        window = from_bounds(*bounds, transform=src.transform)
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        
        # Create mask where values > threshold
        mask = data > threshold
        
        # Vectorize the mask
        geoms = []
        for geom, value in shapes(data, mask=mask, connectivity=4, transform=transform):
            if value > threshold:
                geoms.append({
                    'geometry': shape(geom),
                    'crs': src.crs
                })
        
        return geoms


def intersect_geometry_with_rasters(
    raster_paths: list[str],
    input_gdf: gpd.GeoDataFrame,
    threshold: int = 0
) -> dict:
    """Intersect a WGS84 geometry with the union of pixels (value > threshold) from multiple rasters.

    Args:
        raster_paths: List of paths to COG files (gs://bucket/file.tif for GCS)
        input_gdf: GeoDataFrame with input geometry and properties in WGS84
        threshold: Pixel value threshold (default: 0)
        
    Returns:
        Dictionary with 'geometry' (in WGS84) and 'area' keys
    """
    # Get the union of all input geometries
    input_union = input_gdf.unary_union
    
    # Collect all geometries from all rasters
    all_geoms = []
    target_crs = None
    
    # Get input geometry bounds for filtering
    input_bounds_wgs84 = input_union.bounds  # (minx, miny, maxx, maxy)
    
    for raster_path in raster_paths:
        # Use /vsigs/ prefix for GCS files
        raster_path_gdal = '/vsigs/' + raster_path[5:] if raster_path.startswith('gs://') else raster_path
        
        with rasterio.open(raster_path_gdal) as src:
            if target_crs is None:
                target_crs = src.crs
            
            # Transform input bounds to raster CRS
            input_bounds_transformed = transform_geom(
                f'EPSG:{WGS84_EPSG}',
                src.crs,
                mapping(box(*input_bounds_wgs84))
            )
            input_bbox_raster_crs = shape(input_bounds_transformed).bounds
            raster_bounds = src.bounds
            
            # Check if bounding boxes intersect
            if not (input_bbox_raster_crs[0] <= raster_bounds[2] and
                    input_bbox_raster_crs[2] >= raster_bounds[0] and
                    input_bbox_raster_crs[1] <= raster_bounds[3] and
                    input_bbox_raster_crs[3] >= raster_bounds[1]):
                print(f"Skipping {raster_path} (no spatial overlap)")
                continue
        
        print(f"Processing: {raster_path}")
        
        # Read only the portion that overlaps with input geometry
        geoms = vectorize_raster_mask(raster_path, threshold, bounds=input_bbox_raster_crs)
        
        if geoms:
            all_geoms.extend([g['geometry'] for g in geoms])
    
    if not all_geoms:
        return {
            'geometry': None,
            'area': 0,
            'message': 'No pixels found with value > threshold'
        }
    
    print(f"Found {len(all_geoms)} geometries across all rasters")
    
    # Union all geometries from all rasters
    print("Creating union of all geometries...")
    raster_union = unary_union(all_geoms)
    
    # Transform input geometry from WGS84 to raster CRS
    input_geom_transformed = shape(transform_geom(
        f'EPSG:{WGS84_EPSG}',
        target_crs,
        mapping(input_union)
    ))
    
    # Perform intersection
    print("Computing intersection...")
    intersection = raster_union.intersection(input_geom_transformed)
    
    # Transform result back to WGS84
    result_wgs84 = shape(transform_geom(
        target_crs,
        f'EPSG:{WGS84_EPSG}',
        mapping(intersection)
    ))
    
    return {
        'geometry': mapping(result_wgs84),
        'area': result_wgs84.area,
        'geometry_type': result_wgs84.geom_type
    }


def intersect_with_geopandas(
    raster_paths: list[str],
    input_gdf: gpd.GeoDataFrame,
    threshold: int = 0
) -> gpd.GeoDataFrame:
    """Implementation using GeoPandas for easier manipulation.
    
    Args:
        raster_paths: List of paths to COG files (gs://bucket/file.tif for GCS)
        input_gdf: GeoDataFrame with input geometry and properties in WGS84
        threshold: Pixel value threshold (default: 0)
    
    Returns:
        GeoDataFrame with intersection result in WGS84, preserving input properties
    """
    # Get the union of all input geometries and their bounds
    input_union = input_gdf.unary_union
    input_bounds_wgs84 = input_union.bounds
    
    all_geoms = []
    target_crs = None
    
    for raster_path in raster_paths:
        # Use /vsigs/ prefix for GCS files
        raster_path_gdal = '/vsigs/' + raster_path[5:] if raster_path.startswith('gs://') else raster_path
        
        with rasterio.open(raster_path_gdal) as src:
            if target_crs is None:
                target_crs = src.crs
            
            # Transform input bounds to raster CRS
            input_bounds_transformed = transform_geom(
                f'EPSG:{WGS84_EPSG}',
                src.crs,
                mapping(box(*input_bounds_wgs84))
            )
            input_bbox_raster_crs = shape(input_bounds_transformed).bounds
            raster_bounds = src.bounds
            
            # Check if bounding boxes intersect
            if not (input_bbox_raster_crs[0] <= raster_bounds[2] and
                    input_bbox_raster_crs[2] >= raster_bounds[0] and
                    input_bbox_raster_crs[1] <= raster_bounds[3] and
                    input_bbox_raster_crs[3] >= raster_bounds[1]):
                continue
        
        # Read only the portion that overlaps with input geometry
        geoms = vectorize_raster_mask(raster_path, threshold, bounds=input_bbox_raster_crs)
        
        if geoms:
            all_geoms.extend([g['geometry'] for g in geoms])
    
    if not all_geoms:
        return gpd.GeoDataFrame()
    
    # Create GeoDataFrame with all geometries
    gdf_rasters = gpd.GeoDataFrame(geometry=all_geoms, crs=target_crs)
    
    # Union all geometries
    union_geom = gdf_rasters.unary_union
    gdf_union = gpd.GeoDataFrame(geometry=[union_geom], crs=target_crs)
    
    # Reproject input to match rasters
    gdf_input_reprojected = input_gdf.to_crs(target_crs)
    
    # Perform intersection - this preserves properties from input_gdf
    gdf_result = gpd.overlay(gdf_input_reprojected, gdf_union, how='intersection')
    
    # Transform back to WGS84
    gdf_result = gdf_result.to_crs(f'EPSG:{WGS84_EPSG}')
    
    return gdf_result


def parse_geometry(geom_file: str) -> gpd.GeoDataFrame:
    """Parse geometry from a file.
    
    Args:
        geom_file: Path to geometry file (GeoJSON, Shapefile, GeoPackage, etc.)
        
    Returns:
        GeoDataFrame with geometry and properties in WGS84
    """
    geom_path = Path(geom_file)
    if not geom_path.exists():
        raise FileNotFoundError(f"Geometry file not found: {geom_file}")
    
    # Read with geopandas to preserve all properties
    gdf = gpd.read_file(geom_path)
    
    # Ensure it's in WGS84
    if gdf.crs != f'EPSG:{WGS84_EPSG}':
        gdf = gdf.to_crs(f'EPSG:{WGS84_EPSG}')
    
    return gdf


def main() -> None:
    """Main function for raster-geometry intersection."""
    parser = argparse.ArgumentParser(
        description="Intersect a WGS84 geometry with the union of pixels (value > threshold) from multiple COG rasters"
    )
    parser.add_argument(
        "--raster_files",
        type=str,
        nargs='+',
        required=True,
        help="List of raster paths (gs://bucket/file.tif)",
    )
    parser.add_argument(
        "--geometry_file",
        type=str,
        required=True,
        help="Path to input geometry file in WGS84 (GeoJSON, Shapefile, GeoPackage, etc.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output GeoJSON file",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Pixel value threshold (default: 0). Only pixels with value > threshold are included",
    )
    
    args = parser.parse_args()
    
    # Parse inputs
    print(f"Found {len(args.raster_files)} raster(s)")
    
    print("Parsing input geometry...")
    input_gdf = parse_geometry(args.geometry_file)
    print(f"Loaded {len(input_gdf)} feature(s)")
    print(f"Geometry type: {input_gdf.geometry.iloc[0].geom_type if len(input_gdf) > 0 else 'N/A'}")
    
    # Set output directory
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nProcessing intersection...")
    print(f"  Threshold: {args.threshold}")
    
    # Perform intersection using GeoPandas approach
    result_gdf = intersect_with_geopandas(
        args.raster_files,
        input_gdf,
        threshold=args.threshold
    )
    
    if result_gdf.empty:
        print("\nNo intersection found!")
        print("Either no pixels met the threshold criteria or there was no spatial overlap.")
        return
    
    # Save result as GeoJSON
    print(f"\nSaving result to: {output_path}")
    result_gdf.to_file(output_path, driver="GeoJSON")
    
    # Print summary
    print("\n" + "="*50)
    print("RESULT SUMMARY")
    print("="*50)
    print(f"Number of features: {len(result_gdf)}")
    print(f"Geometry type: {result_gdf.geometry.iloc[0].geom_type if len(result_gdf) > 0 else 'N/A'}")
    print(f"Total area (WGS84 degrees^2): {result_gdf.geometry.area.sum():.6f}")
    print(f"Bounds (WGS84): {result_gdf.total_bounds}")
    print(f"Output saved to: {output_path}")
    print("="*50)


if __name__ == "__main__":
    main()
