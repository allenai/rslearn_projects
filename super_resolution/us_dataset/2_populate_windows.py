import json
import os

from rasterio import CRS
import tqdm


GROUP = "default"


with open('continental_us_utm_tiles.json', 'r') as f:
    data = json.load(f)

for epsg_code, col, row in tqdm.tqdm(data):
    # In 1_compute_us_tiles we should have divided by -METERS_PER_PIXEL.
    # It takes long time to run so here we fix the row.
    row = -row + 1

    crs = CRS.from_epsg(epsg_code)
    name = "{}_{}_{}".format(epsg_code, col, row)
    metadata = {
        "name": name,
        "group": GROUP,
        "projection": {
            "crs": crs.to_string(),
            "x_resolution": 10,
            "y_resolution": -10,
        },
        "bounds": [col*512, row*512, (col+1)*512, (row+1)*512],
        "time_range": ["2019-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"],
        "layer_datas": {},
        "options": {},
    }
    window_dir = os.path.join("/mnt/data/rslearn_superres_dataset/windows/", GROUP, name)
    os.makedirs(window_dir, exist_ok=True)
    with open(os.path.join(window_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
