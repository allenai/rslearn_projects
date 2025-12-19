"""Get 5x5 degree tiles that are not on land."""

import json
from pathlib import Path

from global_land_mask import globe

AOI_DIR = "/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/aois/"
GRID_SIZE = 5

if __name__ == "__main__":
    box = (-30, -40, 70, 35)

    for lon in range(box[0], box[2], GRID_SIZE):
        for lat in range(box[1], box[3], GRID_SIZE):
            coordinates = [
                (lon, lat),
                (lon, lat + GRID_SIZE),
                (lon + GRID_SIZE, lat + GRID_SIZE),
                (lon + GRID_SIZE, lat),
                (lon, lat),
            ]
            # Make sure at least one corner is in the ocean, otherwise we skip this tile.
            at_least_one_water = False
            for coord in coordinates:
                if globe.is_land(coord[1], coord[0]):
                    continue
                at_least_one_water = True

            print(lon, lat, at_least_one_water)

            if not at_least_one_water:
                continue

            fname = Path(AOI_DIR) / f"aoi_{lon}_{lat}_{lon+GRID_SIZE}_{lat+GRID_SIZE}.geojson"
            with fname.open("w") as f:
                feat = {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coordinates],
                    }
                }
                json.dump({
                    "type": "FeatureCollection",
                    "properties": {},
                    "features": [feat],
                }, f)
