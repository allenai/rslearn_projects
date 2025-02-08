"""Test Satlas post-processing."""

import json
import pathlib

import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry

from rslp.satlas.postprocess import merge_points, smooth_points
from rslp.satlas.predict_pipeline import PATCH_SIZE, Application


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
        # Make sure crs property is set correctly.
        feature_projections = [feat["properties"]["projection"] for feat in features]
        feature_projections.sort()
        assert feature_projections == [
            str(proj32601.crs),
            str(proj32601.crs),
            str(proj32602.crs),
        ]


class TestSmoothPoints:
    """Test the smooth_points function."""

    # Minimum number of timesteps for a point to be considered.
    # This is used in smooth_point_labels_viterbi.go to discard regions that are only
    # covered by satellite imagery for a couple months over multi-year time range.
    MIN_VALID_TIMESTEPS = 8

    def make_merge_output(
        self,
        fname: pathlib.Path,
        geometries: list[STGeometry],
        additional_valid_patches: list[tuple[Projection, int, int]] = [],
        scores: list[float] | None = None,
    ) -> None:
        """Make a GeoJSON matching those produced by merge_points.

        This is the input to smooth_points.

        Args:
            fname: the filename to write to.
            geometries: list of STGeometry to include. The valid patches will be set
                automatically to include all of these geometries.
            additional_valid_patches: additional valid patches (besides those covered
                by the geometries).
            scores: optional list of scores of each geometry. If set, it should be the
                same length as geometries.
        """
        # Convert features to GeoJSON.
        features = []
        valid_patches: dict[str, set[tuple[int, int]]] = {}
        for geometry_idx, geometry in enumerate(geometries):
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            crs_str = str(geometry.projection.crs)
            score = scores[geometry_idx] if scores else 1
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "category": "placeholder",
                        "score": score,
                        "projection": crs_str,
                        "col": int(geometry.shp.x),
                        "row": int(geometry.shp.y),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [wgs84_geometry.shp.x, wgs84_geometry.shp.y],
                    },
                }
            )
            if crs_str not in valid_patches:
                valid_patches[crs_str] = set()
            valid_patches[crs_str].add(
                (int(geometry.shp.x) // PATCH_SIZE, int(geometry.shp.y) // PATCH_SIZE)
            )

        for projection, col, row in additional_valid_patches:
            crs_str = str(projection.crs)
            if crs_str not in valid_patches:
                valid_patches[crs_str] = set()
            valid_patches[crs_str].add((col, row))

        # Make and write the FeatureCollection.
        fc = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "valid_patches": {
                    crs_str: list(patch_set)
                    for crs_str, patch_set in valid_patches.items()
                },
            },
        }
        fname.parent.mkdir(parents=True, exist_ok=True)
        with open(fname, "w") as f:
            json.dump(fc, f)

    def test_smooth_nms(self, tmp_path: pathlib.Path) -> None:
        """Verify that smoothing deletes redundant points across projections."""
        # (-174, 0) is on the border between EPSG:32601 and EPSG:32602.
        # So we add it in both projections and make sure smooth_points deletes it.
        wgs84_geom = STGeometry(WGS84_PROJECTION, shapely.Point(-174, 0), None)
        proj32601 = Projection(CRS.from_epsg(32601), 10, -10)
        proj32602 = Projection(CRS.from_epsg(32602), 10, -10)
        geom32601 = wgs84_geom.to_projection(proj32601)
        geom32602 = wgs84_geom.to_projection(proj32602)

        merged_path = tmp_path / "merged"
        smoothed_path = tmp_path / "smooth"

        # Create the input files.
        # We need MIN_VALID_TIMESTEPS files for any points to be produced.
        labels = [f"0000-0{timestep}" for timestep in range(self.MIN_VALID_TIMESTEPS)]
        for label in labels:
            self.make_merge_output(
                merged_path / f"{label}.geojson",
                [geom32601, geom32602],
                scores=[0.5, 0.6],
            )

        # Run the smoothing.
        smooth_points(
            Application.MARINE_INFRA,
            labels[-1],
            str(merged_path),
            str(smoothed_path),
        )

        # Verify the output.
        smoothed_fname = smoothed_path / f"{labels[-1]}.geojson"
        with smoothed_fname.open() as f:
            fc = json.load(f)
        assert len(fc["features"]) == 1

    def test_discard_single_timestep_positive(self, tmp_path: pathlib.Path) -> None:
        """Ignore a point of it is only detected in one timestep."""
        projection = Projection(CRS.from_epsg(32601), 10, -10)
        geometry1 = STGeometry(projection, shapely.Point(0, 0), None)
        geometry2 = STGeometry(
            projection, shapely.Point(PATCH_SIZE // 2, PATCH_SIZE // 2), None
        )

        merged_path = tmp_path / "merged"
        smoothed_path = tmp_path / "smooth"

        # Create the input files.
        # We need MIN_VALID_TIMESTEPS files where the patch is valid, otherwise it will
        # be ignored.
        # We use two points here since timesteps without any points yield no output.
        # It also makes it easier to ensure that patch is marked valid.
        labels = [f"0000-0{timestep}" for timestep in range(self.MIN_VALID_TIMESTEPS)]
        for timestep, label in enumerate(labels):
            if timestep == 4:
                self.make_merge_output(
                    merged_path / f"{label}.geojson",
                    [geometry1, geometry2],
                )
            else:
                self.make_merge_output(
                    merged_path / f"{label}.geojson",
                    [geometry2],
                )

        # Run the smoothing.
        smooth_points(
            Application.MARINE_INFRA,
            labels[-1],
            str(merged_path),
            str(smoothed_path),
        )

        # Verify the output.
        smoothed_fname = smoothed_path / f"{labels[-1]}.geojson"
        with smoothed_fname.open() as f:
            fc = json.load(f)
        assert len(fc["features"]) == 1

    def test_invalid_timesteps_ignored(self, tmp_path: pathlib.Path) -> None:
        """Ensure a point keeps being predicted if its patch is not observed."""
        projection = Projection(CRS.from_epsg(32601), 10, -10)
        geometry = STGeometry(projection, shapely.Point(0, 0), None)

        merged_path = tmp_path / "merged"
        smoothed_path = tmp_path / "smooth"

        # Create the input files.
        # In the first timesteps, we detect the point.
        labels = [
            f"0000-0{timestep}" for timestep in range(self.MIN_VALID_TIMESTEPS * 2)
        ]
        for label in labels[0 : self.MIN_VALID_TIMESTEPS]:
            self.make_merge_output(
                merged_path / f"{label}.geojson",
                [geometry],
            )
        # In the remaining timesteps, we leave the patch invalid.
        for label in labels[self.MIN_VALID_TIMESTEPS :]:
            self.make_merge_output(
                merged_path / f"{label}.geojson",
                [],
            )

        # Run the smoothing.
        smooth_points(
            Application.MARINE_INFRA,
            labels[-1],
            str(merged_path),
            str(smoothed_path),
        )

        # Verify the output.
        smoothed_fname = smoothed_path / f"{labels[-1]}.geojson"
        with smoothed_fname.open() as f:
            fc = json.load(f)
        assert len(fc["features"]) == 1

    def test_point_removed(self, tmp_path: pathlib.Path) -> None:
        """Verify that a point will be removed with enough negative observations."""
        projection = Projection(CRS.from_epsg(32601), 10, -10)
        geometry1 = STGeometry(projection, shapely.Point(0, 0), None)
        # As with test_discard_single_timestep_positive, we use a second point so it is
        # easier to ensure the patch is valid and so that outputs are produced at all
        # timesteps. Here, we will observe geometry2 at every timestep but cut
        # geometry1 off halfway through.
        geometry2 = STGeometry(
            projection, shapely.Point(PATCH_SIZE // 2, PATCH_SIZE // 2), None
        )

        merged_path = tmp_path / "merged"
        smoothed_path = tmp_path / "smooth"

        # We will have 30 positive observations and then 30 negative ones.
        # This ensures enough because there is very low transition probability from
        # positive to negative (since it is unlikely that e.g. a wind turbine would be
        # torn down).
        num_observations = 30

        # Create the input files.
        # In the first timesteps, we detect both points.
        labels = [f"0000-0{timestep}" for timestep in range(num_observations * 2)]
        for label in labels[0:num_observations]:
            self.make_merge_output(
                merged_path / f"{label}.geojson",
                [geometry1, geometry2],
            )
        # In the remaining timesteps, we only detect geometry2.
        for label in labels[num_observations:]:
            self.make_merge_output(
                merged_path / f"{label}.geojson",
                [geometry2],
            )

        # Run the smoothing.
        smooth_points(
            Application.MARINE_INFRA,
            labels[-1],
            str(merged_path),
            str(smoothed_path),
        )

        # At the first timestep, we should see both points.
        smoothed_fname = smoothed_path / f"{labels[0]}.geojson"
        with smoothed_fname.open() as f:
            fc = json.load(f)
        assert len(fc["features"]) == 2

        # At the last timestep, we should only see geometry2.
        smoothed_fname = smoothed_path / f"{labels[-1]}.geojson"
        with smoothed_fname.open() as f:
            fc = json.load(f)
        assert len(fc["features"]) == 1
