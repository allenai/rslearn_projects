"""Test rslp.satlas.postprocess_raster."""

import json
import pathlib

import numpy as np
import numpy.typing as npt
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from rslp.satlas.postprocess_raster import smooth_rasters
from rslp.satlas.predict_pipeline import Application, get_output_fname


class TestSmoothRasters:
    """Test the smooth_rasters function."""

    def get_smoothed_output_with_inputs(
        self, scratch_dir: UPath, arrays: list[npt.NDArray]
    ) -> list[npt.NDArray]:
        """Run smoothing with the given input rasters.

        This function takes care of placing them with dummy timesteps into a prediction
        output directory, running smooth_rasters, and then collecting the outputs.
        """
        height, width = arrays[0].shape[1:3]
        bounds = (0, 0, width, height)
        projection = WGS84_PROJECTION
        predict_path = scratch_dir / "predict"
        smoothed_path = scratch_dir / "smoothed"

        # Assign dummy timestep to each raster.
        label_options = [f"2024-{month:02d}" for month in range(1, 13)]
        assert len(arrays) <= len(label_options)
        labels = label_options[0 : len(arrays)]

        # Write the input rasters.
        for array, label in zip(arrays, labels):
            fname = get_output_fname(
                Application.SOLAR_FARM,
                (predict_path / label).as_uri(),
                projection,
                bounds,
            )
            GeotiffRasterFormat().encode_raster(
                fname.parent,
                projection,
                bounds,
                RasterArray(chw_array=array),
                fname=fname.name,
            )

        # Apply smoothing.
        # Note that only the first NUM_HISTORICAL_TIMESTEPS rasters will be smoothed.
        smooth_rasters(
            application=Application.SOLAR_FARM,
            label=labels[-1],
            predict_path=predict_path.as_uri(),
            smoothed_path=smoothed_path.as_uri(),
            projection_json=json.dumps(projection.serialize()),
            bounds=bounds,
        )

        # Read the smoothed outputs.
        outputs = []
        for label in labels:
            fname = get_output_fname(
                Application.SOLAR_FARM,
                (smoothed_path / label).as_uri(),
                projection,
                bounds,
            )
            raster = GeotiffRasterFormat().decode_raster(
                fname.parent, projection, bounds, fname=fname.name
            )
            outputs.append(raster.get_chw_array())

        return outputs

    def test_single_timestep(self, tmp_path: pathlib.Path) -> None:
        """With single timestep, it should respect that one raster."""
        array = np.zeros((1, 2, 2), dtype=np.uint8)
        array[0, 0, 0] = 2
        outputs = self.get_smoothed_output_with_inputs(UPath(tmp_path), [array])
        print(outputs[0], array)
        assert np.all(outputs[0] == array)

    def test_false_positive(self, tmp_path: pathlib.Path) -> None:
        """Make sure one timestep with detection doesn't make it positive."""
        negative_array = np.ones((1, 1, 1), dtype=np.uint8)
        positive_array = np.ones((1, 1, 1), dtype=np.uint8) * 2
        outputs = self.get_smoothed_output_with_inputs(
            UPath(tmp_path),
            [
                negative_array,
                negative_array,
                positive_array,
                negative_array,
                negative_array,
            ],
        )
        print(outputs)
        assert all(np.all(array == 1) for array in outputs)
