"""Geospatial intersection utilities for COG rasters and vector geometries.

This module provides functions to efficiently intersect vector geometries with Cloud Optimized GeoTIFF (COG) rasters.
"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.session import GSSession
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds
from rslearn.const import WGS84_EPSG
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union


# Configure GDAL for efficient COG access
def configure_gdal_for_cogs() -> None:
    """Configure GDAL environment variables for optimal COG performance."""
    # Enable HTTP range requests
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff"
    # Cache settings for better performance
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["CPL_VSIL_CURL_CACHE_SIZE"] = str(256 * 2**20)  # 256 MB


def setup_gcs_authentication(raster_paths: list[str]) -> None:
    """Set up Google Cloud Storage authentication if needed.

    Sets GOOGLE_APPLICATION_CREDENTIALS environment variable if:
    1. The env var isn't already set
    2. The default credentials file exists
    3. Any raster files are on GCS (gs:// URLs)

    Args:
        raster_paths: List of raster file paths to check for GCS URLs
    """
    # Check if any raster files are on GCS
    has_gcs_files = any(path.startswith("gs://") for path in raster_paths)

    if not has_gcs_files:
        return

    # Check if GOOGLE_APPLICATION_CREDENTIALS is already set
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    # Check if default credentials file exists
    default_creds_path = (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    )
    if not default_creds_path.exists():
        print(
            f"Warning: GCS files detected but no credentials found at {default_creds_path}"
        )
        print("Run 'gcloud auth application-default login' to set up authentication")
        return

    # Set the environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(default_creds_path)
    print(f"Set GOOGLE_APPLICATION_CREDENTIALS to {default_creds_path}")


configure_gdal_for_cogs()


def vectorize_raster_mask(
    raster_path: str, threshold: int, bounds: tuple[float, float, float, float]
) -> list[dict]:
    """Vectorize pixels from a raster where value > threshold.

    Args:
        raster_path: Path to GeoTIFF (gs://bucket/file.tif for GCS)
        threshold: Pixel value threshold (default: 0)
        bounds: Optional bounding box (left, bottom, right, top) in raster CRS
                to limit reading. When provided, only this window is read (efficient with COGs).

    Returns:
        List of geometry dictionaries with CRS info
    """
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
                geoms.append({"geometry": shape(geom), "crs": src.crs})

        return geoms


def intersect_geometry_with_rasters(
    raster_paths: list[str],
    input_gdf: gpd.GeoDataFrame,
    threshold: int = 0,
    simplify_tolerance: float | None = None,
) -> dict:
    """Intersect a WGS84 geometry with the union of pixels (value > threshold) from multiple rasters.

    Args:
        raster_paths: List of paths to COG files (gs://bucket/file.tif for GCS)
        input_gdf: GeoDataFrame with input geometry and properties in WGS84
        threshold: Pixel value threshold (default: 0)
        simplify_tolerance: Optional tolerance for geometry simplification using Douglas-Peucker algorithm.
                          Higher values result in more simplified geometries. If None, no simplification is applied.

    Returns:
        Dictionary with 'geometry' (in WGS84) and 'area' keys
    """
    # Get the union of all input geometries
    input_union = input_gdf.union_all()

    # Collect all geometries from all rasters
    all_geoms = []
    target_crs = None

    # Get input geometry bounds for filtering
    input_bounds_wgs84 = input_union.bounds  # (minx, miny, maxx, maxy)

    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            if target_crs is None:
                target_crs = src.crs

            # Transform input bounds to raster CRS
            input_bounds_transformed = transform_geom(
                f"EPSG:{WGS84_EPSG}", src.crs, mapping(box(*input_bounds_wgs84))
            )
            input_bbox_raster_crs = shape(input_bounds_transformed).bounds
            raster_bounds = src.bounds

            # Check if bounding boxes intersect
            if not (
                input_bbox_raster_crs[0] <= raster_bounds[2]
                and input_bbox_raster_crs[2] >= raster_bounds[0]
                and input_bbox_raster_crs[1] <= raster_bounds[3]
                and input_bbox_raster_crs[3] >= raster_bounds[1]
            ):
                print(f"Skipping {raster_path} (no spatial overlap)")
                continue

        print(f"Processing: {raster_path}")

        # Read only the portion that overlaps with input geometry
        geoms = vectorize_raster_mask(
            raster_path, threshold, bounds=input_bbox_raster_crs
        )

        if geoms:
            all_geoms.extend([g["geometry"] for g in geoms])

    if not all_geoms:
        return {
            "geometry": None,
            "area": 0,
            "message": "No pixels found with value > threshold",
        }

    print(f"Found {len(all_geoms)} geometries across all rasters")

    # Union all geometries from all rasters
    print("Creating union of all geometries...")
    raster_union = unary_union(all_geoms)

    # Transform input geometry from WGS84 to raster CRS
    input_geom_transformed = shape(
        transform_geom(f"EPSG:{WGS84_EPSG}", target_crs, mapping(input_union))
    )

    # Perform intersection
    print("Computing intersection...")
    intersection = raster_union.intersection(input_geom_transformed)

    # Transform result back to WGS84
    result_wgs84 = shape(
        transform_geom(target_crs, f"EPSG:{WGS84_EPSG}", mapping(intersection))
    )

    # Apply geometry simplification if tolerance is specified
    if simplify_tolerance is not None:
        print(f"Simplifying geometry with tolerance: {simplify_tolerance}")
        result_wgs84 = result_wgs84.simplify(
            tolerance=simplify_tolerance, preserve_topology=True
        )

    return {
        "geometry": mapping(result_wgs84),
        "area": result_wgs84.area,
        "geometry_type": result_wgs84.geom_type,
    }


def intersect_with_geopandas(
    raster_paths: list[str],
    input_gdf: gpd.GeoDataFrame,
    threshold: int = 0,
    simplify_tolerance: float | None = None,
) -> gpd.GeoDataFrame:
    """Implementation using GeoPandas for easier manipulation.

    Args:
        raster_paths: List of paths to COG files (gs://bucket/file.tif for GCS)
        input_gdf: GeoDataFrame with input geometry and properties in WGS84
        threshold: Pixel value threshold (default: 0)
        simplify_tolerance: Optional tolerance for geometry simplification using Douglas-Peucker algorithm.
                          Higher values result in more simplified geometries. If None, no simplification is applied.

    Returns:
        GeoDataFrame with intersection result in WGS84, preserving input properties
    """
    # Get the union of all input geometries and their bounds
    input_union = input_gdf.union_all()
    input_bounds_wgs84 = input_union.bounds

    all_geoms = []
    target_crs = None

    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            if target_crs is None:
                target_crs = src.crs

            # Transform input bounds to raster CRS
            input_bounds_transformed = transform_geom(
                f"EPSG:{WGS84_EPSG}", src.crs, mapping(box(*input_bounds_wgs84))
            )
            input_bbox_raster_crs = shape(input_bounds_transformed).bounds
            raster_bounds = src.bounds

            # Check if bounding boxes intersect
            if not (
                input_bbox_raster_crs[0] <= raster_bounds[2]
                and input_bbox_raster_crs[2] >= raster_bounds[0]
                and input_bbox_raster_crs[1] <= raster_bounds[3]
                and input_bbox_raster_crs[3] >= raster_bounds[1]
            ):
                continue

            # Read only the portion that overlaps with input geometry
            geoms = vectorize_raster_mask(
                raster_path, threshold, bounds=input_bbox_raster_crs
            )

            if geoms:
                all_geoms.extend([g["geometry"] for g in geoms])

    if not all_geoms:
        return gpd.GeoDataFrame()

    # Create GeoDataFrame with all geometries
    gdf_rasters = gpd.GeoDataFrame(geometry=all_geoms, crs=target_crs)

    # Union all geometries
    union_geom = gdf_rasters.union_all()
    gdf_union = gpd.GeoDataFrame(geometry=[union_geom], crs=target_crs)

    # Reproject input to match rasters
    gdf_input_reprojected = input_gdf.to_crs(target_crs)

    # Perform intersection - this preserves properties from input_gdf
    gdf_result = gpd.overlay(gdf_input_reprojected, gdf_union, how="intersection")

    # Transform back to WGS84
    gdf_result = gdf_result.to_crs(f"EPSG:{WGS84_EPSG}")

    # Apply geometry simplification if tolerance is specified
    if simplify_tolerance is not None and not gdf_result.empty:
        print(f"Simplifying geometries with tolerance: {simplify_tolerance}")
        gdf_result.geometry = gdf_result.geometry.simplify(
            tolerance=simplify_tolerance, preserve_topology=True
        )

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
    if gdf.crs != f"EPSG:{WGS84_EPSG}":
        gdf = gdf.to_crs(f"EPSG:{WGS84_EPSG}")

    return gdf


def main() -> None:
    """Main function for raster-geometry intersection."""
    parser = argparse.ArgumentParser(
        description="Intersect a WGS84 geometry with the union of pixels (value > threshold) from multiple COG rasters"
    )
    parser.add_argument(
        "--raster_files",
        type=str,
        nargs="+",
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
    parser.add_argument(
        "--simplify_tolerance",
        type=float,
        default=None,
        help="Optional tolerance for geometry simplification using Douglas-Peucker algorithm. "
        "Higher values result in more simplified geometries. If not specified, no simplification is applied. "
        "Example values: 0.0001 for minimal simplification, 0.001 for moderate, 0.01 for aggressive.",
    )

    args = parser.parse_args()

    # Set up GCS authentication if needed
    setup_gcs_authentication(args.raster_files)

    print(f"Found {len(args.raster_files)} raster(s)")
    print("Parsing input geometry...")
    input_gdf = parse_geometry(args.geometry_file)
    print(f"Loaded {len(input_gdf)} feature(s)")
    print(
        f"Geometry type: {input_gdf.geometry.iloc[0].geom_type if len(input_gdf) > 0 else 'N/A'}"
    )

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nProcessing intersection...")
    print(f"  Pixel value threshold: {args.threshold}")
    if args.simplify_tolerance is not None:
        print(f"  Simplification tolerance: {args.simplify_tolerance}")

    with rasterio.Env(GSSession()):
        result_gdf = intersect_with_geopandas(
            args.raster_files,
            input_gdf,
            threshold=args.threshold,
            simplify_tolerance=args.simplify_tolerance,
        )

        if result_gdf.empty:
            print(
                "No intersection found! Either no pixels met the threshold criteria or there was no spatial overlap."
            )
            return

        # Save result as GeoJSON
        print(f"\nSaving result to: {output_path}")
        result_gdf.to_file(output_path, driver="GeoJSON")


if __name__ == "__main__":
    main()
