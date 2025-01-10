"""Tool to create smaller cropped tiffs for testing."""

import argparse

import rasterio
from rasterio.windows import Window

from rslp.log_utils import get_logger

logger = get_logger(__name__)


def crop_geotiff(
    input_path: str,
    output_path: str,
    x_start: int,
    y_start: int,
    width: int,
    height: int,
) -> None:
    """Crop a geotiff and save the result to a new file."""
    with rasterio.open(input_path) as src:
        # Create a window for cropping
        window = Window(x_start, y_start, width, height)

        # Read the data through the window
        data = src.read(window=window)

        # Update the transform for the new cropped image
        transform = rasterio.windows.transform(window, src.transform)

        # Update profile for the new file
        profile = src.profile.copy()
        logger.info(f"{transform=}")
        profile.update({"height": height, "width": width, "transform": transform})

        # Write the cropped image
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop a GeoTIFF file to specified dimensions"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input GeoTIFF file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save cropped GeoTIFF file",
    )
    parser.add_argument(
        "--x-start",
        type=int,
        required=True,
        help="Starting x coordinate for crop window",
    )
    parser.add_argument(
        "--y-start",
        type=int,
        required=True,
        help="Starting y coordinate for crop window",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Width of crop window in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Height of crop window in pixels",
    )

    args = parser.parse_args()

    crop_geotiff(
        args.input,
        args.output,
        args.x_start,
        args.y_start,
        args.width,
        args.height,
    )
