"""Convert the Sentinel-2 vessel labels in siv to rslearn format

Also switch to using UTM projection.

Since the labels are non-static, we write the image ID too, and there is another script
that runs after `dataset prepare` to delete non-matching images from items.json.
"""

import multiprocessing
import sqlite3
from datetime import datetime, timedelta, timezone

import shapely
import tqdm

from ..lib import convert_window

db_path = "/data/favyenb/sentinel2-vessel-data/metadata.sqlite3"
out_dir = "/data/favyenb/rslearn_datasets_satlas/sentinel2_vessel/"

conn = sqlite3.connect(db_path)
conn.isolation_level = None
db = conn.cursor()

# Cache the labels first.
db.execute("""
    SELECT l.window_id, l.column, l.row
    FROM labels AS l, windows AS w
    WHERE l.window_id = w.id AND w.dataset_id = 3
""")
window_to_labels = {}
for w_id, l_col, l_row in tqdm.tqdm(db.fetchall(), desc="Read labels"):
    if w_id not in window_to_labels:
        window_to_labels[w_id] = []
    window_to_labels[w_id].append((l_col, l_row))

# Get the windows.
db.execute("""
    SELECT w.id, im.name, w.column, w.row, w.width, w.height, w.split
    FROM windows AS w, images AS im
    WHERE w.dataset_id = 3 AND w.image_id = im.id AND w.width >= 512 AND w.height >= 512
""")
convert_window_jobs = []
for w_id, im_name, w_col, w_row, w_width, w_height, w_split in tqdm.tqdm(
    db.fetchall(), desc="Read windows"
):
    bounds = [w_col, w_row, w_col + w_width, w_row + w_height]

    parts = im_name.split("_")
    ts_str = parts[2]
    assert len(ts_str) == 15
    ts = datetime(
        year=int(ts_str[0:4]),
        month=int(ts_str[4:6]),
        day=int(ts_str[6:8]),
        hour=int(ts_str[9:11]),
        minute=int(ts_str[11:13]),
        second=int(ts_str[13:15]),
        tzinfo=timezone.utc,
    )
    time_range = (
        ts - timedelta(minutes=5),
        ts + timedelta(minutes=5),
    )

    labels = []
    for l_col, l_row in window_to_labels.get(w_id, []):
        point = shapely.Point(l_col, l_row)
        properties = {"category": "vessel"}
        labels.append((point, properties))

    convert_window_jobs.append(
        dict(
            root_dir=out_dir,
            group=w_split,
            zoom=13,
            bounds=bounds,
            labels=labels,
            time_range=time_range,
            image_name=im_name,
            window_name=f"{w_col}_{w_row}_{w_id}",
        )
    )


def process(job):
    image_name = job.pop("image_name")
    window = convert_window(**job)
    with window.file_api.open("image_name_from_siv.txt", "w") as f:
        f.write(image_name)


p = multiprocessing.Pool(64)
outputs = p.imap_unordered(process, convert_window_jobs)
for _ in tqdm.tqdm(outputs, total=len(convert_window_jobs), desc="Write windows"):
    pass
p.close()
