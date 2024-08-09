"""So Paige said a zoomed out view would be useful.
The default one is 64x64 at 15 m/pixel (after pan-sharpening).
It seems like second image zoomed out 4x and another zoomed out 16x would be useful.
4x means 256x256 at 15 m/pixel.
16x means 1024x1024 at 15 m/pixel.
So we create 1024x1024 windows.
"""

import os

from rslearn.dataset import Window
from rslearn.utils import LocalFileAPI

src_dir = "/data/favyenb/rslearn_landsat/2024-07-18-joe-check-training-phase1/windows/phase2a/"
dst_dir = "/data/favyenb/rslearn_landsat/2024-07-18-joe-check-training-phase1/windows/phase2a_zoomout/"

for window_id in os.listdir(src_dir):
    window = Window.load(LocalFileAPI(os.path.join(src_dir, window_id)))
    center_col = (window.bounds[0] + window.bounds[2]) // 2
    center_row = (window.bounds[1] + window.bounds[3]) // 2
    new_bounds = (
        center_col - 512,
        center_row - 512,
        center_col + 512,
        center_row + 512,
    )
    new_window_id = window_id + "_zoomout"
    new_window_root = os.path.join(dst_dir, window_id + "_zoomout")
    os.makedirs(new_window_root, exist_ok=True)
    new_window = Window(
        file_api=LocalFileAPI(new_window_root),
        group="phase2a_zoomout",
        name=new_window_id,
        projection=window.projection,
        bounds=new_bounds,
        time_range=window.time_range,
    )
    new_window.save()
