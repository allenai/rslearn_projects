"""Test Satlas post-processing."""

import json
import pathlib

from rasterio.crs import CRS
from rslearn.utils.geometry import Projection

from rslp.satlas.postprocess import merge_points
from rslp.satlas.predict_pipeline import Application


class TestMergePoints:
    """Test the merge_points function."""

    def make_task_output(
        self,
        fname: pathlib.Path,
        projection: Projection,
        coords: list[tuple[float, float]],
        valid_patches: list[tuple[int, int]],
    ) -> None:
        """Make a JSON matching those produced by the Satlas predict_pipeline.

        Args:
            fname: the filename to write to.
            projection: the projection of the prediction task. The task writes the
                GeoJSON in pixel coordinates under that projection.
            coords: list of point (col, row) coordinates to include.
            valid_patches: list of (col, row) patches to include. These are in tiles of
                PATCH_SIZE (see rslp.satlas.predict_pipeline).
        """
        # Convert features to GeoJSON.
        features = []
        for col, row in coords:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "score": 1,
                        "category": "placeholder",
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [col, row],
                    },
                }
            )

        # Make the FeatureCollection.
        fc = {
            "type": "FeatureCollection",
            "features": features,
            "properties": projection.serialize(),
        }
        # Add the valid patches. It is a dict from CRS to tile list.
        fc["properties"]["valid_patches"] = {
            str(projection.crs): valid_patches,
        }

        fname.parent.mkdir(parents=True, exist_ok=True)
        with open(fname, "w") as f:
            json.dump(fc, f)

    def test_two_crs_merged(self, tmp_path: pathlib.Path) -> None:
        """Verify that when merging across two CRS it is successful."""
        proj32601 = Projection(CRS.from_epsg(32601), 10, -10)
        proj32602 = Projection(CRS.from_epsg(32602), 10, -10)

        predict_path = tmp_path / "predict"
        merged_path = tmp_path / "merged"

        # File 1 is in 32601 and contains one feature.
        self.make_task_output(predict_path / "1.geojson", proj32601, [(0, 0)], [(0, 0)])
        # File 2 is also in 32601 and contains one different feature and patch.
        self.make_task_output(
            predict_path / "2.geojson", proj32601, [(2048, 2048)], [(1, 1)]
        )
        # File 3 is in 32602 and contains a third feature.
        self.make_task_output(predict_path / "3.geojson", proj32602, [(0, 0)], [(0, 0)])

        # Run the merging.
        merge_points(
            Application.MARINE_INFRA,
            # Use arbitrary YYYY-MM label.
            "1234-56",
            str(predict_path),
            str(merged_path),
        )

        # Verify the output.
        merged_fname = merged_path / "1234-56.geojson"
        with merged_fname.open() as f:
            fc = json.load(f)
        # And the valid patches should be merged, with one in 32601 and two in 32602.
        valid_patches = fc["properties"]["valid_patches"]
        assert len(valid_patches) == 2
        patches32601 = valid_patches[str(proj32601.crs)]
        patches32601.sort()
        assert patches32601 == [[0, 0], [1, 1]]
        patches32602 = valid_patches[str(proj32602.crs)]
        patches32602.sort()
        assert patches32602 == [[0, 0]]

        features = fc["features"]
        assert len(features) == 3
