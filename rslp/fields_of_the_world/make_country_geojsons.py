"""Create GeoJSON files containing the center points of the samples in each country."""

import json
import sys
from pathlib import Path

import geopandas as gpd

if __name__ == "__main__":
    base_dir = sys.argv[
        1
    ]  # /weka/dfive-default/rslearn-eai/artifacts/fields_of_the_world/
    out_dir = sys.argv[2]
    chip_fnames = Path(base_dir).glob("*/chips_*.parquet")
    for chip_fname in chip_fnames:
        country_name = chip_fname.parent.name
        gdf = gpd.read_parquet(chip_fname)
        features = []
        for shp in gdf.geometry:
            features.append(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [shp.centroid.x, shp.centroid.y],
                    },
                }
            )

        with (Path(out_dir) / f"{country_name}.geojson").open("w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "properties": {},
                    "features": features,
                },
                f,
            )
