"""We initially named the tasks differently so we rename it to better format."""

import random
import shutil

import tqdm
from rslearn.dataset.dataset import Dataset, Window
from upath import UPath

DATASET_PATH = "/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/rslearn_dataset_for_selected_events/"


if __name__ == "__main__":
    ds_path = UPath(DATASET_PATH)
    dataset = Dataset(ds_path)
    windows = dataset.load_windows()
    random.shuffle(windows)
    for idx, window in enumerate(tqdm.tqdm(windows)):
        src_name = window.name
        _, lon_str, lat_str, predicted_category = src_name.split("_")
        date_time_str = window.time_range[0].strftime("%Y-%m-%d")
        dst_name = f"[#{idx+1:04d}] {date_time_str} at {float(lat_str):.04f}, {float(lon_str):.04f} prediction:{predicted_category}"
        shutil.move(
            Window.get_window_root(ds_path, window.group, src_name),
            Window.get_window_root(ds_path, window.group, dst_name),
        )
        window.name = dst_name
        window.save()
