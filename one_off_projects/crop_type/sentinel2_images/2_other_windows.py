"""Alternative script for populating windows for AgriFieldNet and South Africa.
We just copy the bounds from the 256x256 label files that YiChia cropped
(after projecting the original 256x256 labels to WebMercator).
"""

import os
from datetime import datetime, timezone

import rasterio
import tqdm
from rslearn.dataset import Window
from rslearn.utils import Projection

out_dir = "/data/favyenb/rslearn_crop_type/windows/"

# in_dir = "/data/yichiac/sact_harmonized/train/labels/"
# group = "sact"
# time_ranges = [
#    ("2017-02", (datetime(2017, 2, 1, tzinfo=timezone.utc), datetime(2017, 3, 1, tzinfo=timezone.utc))),
#    ("2017-03", (datetime(2017, 3, 1, tzinfo=timezone.utc), datetime(2017, 4, 1, tzinfo=timezone.utc))),
#    ("2017-04", (datetime(2017, 4, 1, tzinfo=timezone.utc), datetime(2017, 5, 1, tzinfo=timezone.utc))),
# ]

in_dir = "/data/yichiac/agrifieldnet_harmonized/train_labels/"
group = "agrifieldnet"
time_ranges = [
    (
        "2021-06",
        (
            datetime(2021, 6, 1, tzinfo=timezone.utc),
            datetime(2021, 7, 1, tzinfo=timezone.utc),
        ),
    ),
    (
        "2021-07",
        (
            datetime(2021, 7, 1, tzinfo=timezone.utc),
            datetime(2021, 8, 1, tzinfo=timezone.utc),
        ),
    ),
    (
        "2021-08",
        (
            datetime(2021, 8, 1, tzinfo=timezone.utc),
            datetime(2021, 9, 1, tzinfo=timezone.utc),
        ),
    ),
]

pixel_size = 10

for fname in tqdm.tqdm(os.listdir(in_dir)):
    with rasterio.open(os.path.join(in_dir, fname)) as src:
        assert abs(src.transform.a - pixel_size) < 1e-6
        assert abs(src.transform.e - (-pixel_size)) < 1e-6
        projection = Projection(src.crs, pixel_size, -pixel_size)
        bounds = (
            int(src.bounds.left) // pixel_size,
            int(src.bounds.top) // -pixel_size,
            int(src.bounds.right) // pixel_size,
            int(src.bounds.bottom) // -pixel_size,
        )
        for time_label, time_range in time_ranges:
            window_name = fname.split(".tif")[0] + "_" + time_label
            window = Window(
                window_root=os.path.join(out_dir, group, window_name),
                group=group,
                name=window_name,
                projection=projection,
                bounds=bounds,
                time_range=time_range,
            )
            window.save()
