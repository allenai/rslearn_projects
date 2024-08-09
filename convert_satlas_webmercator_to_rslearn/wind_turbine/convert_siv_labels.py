"""Convert the wind turbine labels in siv to rslearn format while also switching to using
UTM projection.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import shapely

from ..lib import convert_window

db_path = "/home/ubuntu/siv_l1c/data/siv.sqlite3"
out_dir = "/multisat/datasets/rslearn_datasets_satlas/wind_turbine/"
group = "label"

conn = sqlite3.connect(db_path)
conn.isolation_level = None
db = conn.cursor()

# Get the windows.
db.execute("""
    SELECT w.id, im.time, w.column, w.row, w.width, w.height
    FROM windows AS w, images AS im
    WHERE dataset_id = 20 AND w.image_id = im.id
    AND split in ('fp01-done', 'fp02-done', 'fp03-done', 'fp04-auto-done', 'fp05', 'fp04', '2023sep06', 'fp07')
""")
for w_id, im_time, w_col, w_row, w_width, w_height in db.fetchall():
    bounds = [w_col, w_row, w_col + w_width, w_row + w_height]

    ts = datetime.fromisoformat(im_time)
    if not ts.tzinfo:
        ts = ts.replace(tzinfo=timezone.utc)
    time_range = (
        ts - timedelta(days=60),
        ts + timedelta(days=30),
    )

    db.execute(
        """
        SELECT column, row FROM labels WHERE window_id = ?
    """,
        (w_id,),
    )
    labels = []
    for l_col, l_row in db.fetchall():
        point = shapely.Point(l_col, l_row)
        properties = {"category": "turbine"}
        labels.append((point, properties))

    convert_window(
        root_dir=out_dir,
        group=group,
        zoom=13,
        bounds=bounds,
        labels=labels,
        time_range=time_range,
    )
