"""Tool to create smaller cropped tiffs for testing."""

from pathlib import Path

import rasterio
from rasterio.windows import Window


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
        print(transform)
        profile.update({"height": height, "width": width, "transform": transform})

        # Write the cropped image
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)


if __name__ == "__main__":
    original_path = str(
        Path(__file__).parents[1]
        / "test_data/forest_loss_driver/alert_dates/070W_10S_060W_00N.tif"
    )
    cropped_path = str(
        Path(__file__).parents[1]
        / "test_data/forest_loss_driver/alert_dates/cropped_070W_10S_060W_00N.tif"
    )
    x_start = 0
    y_start = 40000
    width = 10000
    height = 10000
    crop_geotiff(original_path, cropped_path, x_start, y_start, width, height)

    original_conf_path = str(
        Path(__file__).parents[1]
        / "test_data/forest_loss_driver/alert_tiffs/070W_10S_060W_00N.tif"
    )
    cropped_conf_path = str(
        Path(__file__).parents[1]
        / "test_data/forest_loss_driver/alert_tiffs/cropped_070W_10S_060W_00N.tif"
    )
    crop_geotiff(original_conf_path, cropped_conf_path, x_start, y_start, width, height)
