from datetime import datetime, timedelta, timezone
import json
import os

from rasterio import CRS
import tqdm


GROUP = "default"
naip_tile_dir = "/mnt/data/rslearn_superres_dataset/tiles/"
sentinel2_window_dir = "/mnt/superres_sentinel2/windows/"


# Set of (CRS, col, row, year, month).
needed = set()

for image_id in tqdm.tqdm(os.listdir(naip_tile_dir), desc="Identifying needed Sentinel-2 tiles"):
    image_dir = os.path.join(naip_tile_dir, image_id)
    parts = image_id.split('_')
    crs_str = parts[-2]
    date_str = parts[-3]
    date = datetime.strptime(date_str, "%Y%m%d")

    # Get year/month within 60 days of this date.
    year_months = set()
    for offset in range(-60, 61, 15):
        offset_date = date + timedelta(days=offset)
        year_months.add((offset_date.year, offset_date.month))

    # Get distinct tiles 3 zooms lower.
    factor = 8
    tiles = set()
    for fname in os.listdir(image_dir):
        if not fname.endswith(".png"):
            continue
        parts = fname.split('.')[0].split('_')
        col = int(parts[0])
        row = int(parts[1])
        tiles.add((col//factor, row//factor))

    for year, month in year_months:
        for tile in tiles:
            needed.add((crs_str, tile[0], tile[1], year, month))


for crs_str, col, row, year, month in tqdm.tqdm(needed, desc="Creating Sentinel-2 windows"):
    name = "{}_{}_{}_{}_{}".format(crs_str, col, row, year, month)

    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1

    time_range = [
        datetime(year, month, 1, tzinfo=timezone.utc).isoformat(),
        datetime(next_year, next_month, 1, tzinfo=timezone.utc).isoformat(),
    ]
    metadata = {
        "name": name,
        "group": GROUP,
        "projection": {
            "crs": crs_str,
            "x_resolution": 10,
            "y_resolution": -10,
        },
        "bounds": [col*512, row*512, (col+1)*512, (row+1)*512],
        "time_range": time_range,
        "options": {},
    }
    window_dir = os.path.join(sentinel2_window_dir, GROUP, name)
    os.makedirs(window_dir, exist_ok=True)
    with open(os.path.join(window_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
