"""
This creates CSV of timestamps, longitudes, and latitudes labeled as vessels.
So I will send that to Hunter and he will get details from AIS then hopefully we can train on all that.

This script works with the siv-l1c sqlite3 db at https://console.cloud.google.com/storage/browser/_details/satlas-explorer-data/siv-annotations/sentinel2.tar
"""
import csv
from datetime import datetime
import math

from siv.db import db

def mercator_to_geo(p, zoom, pixels):
    n = 2**zoom
    x = p[0] / pixels
    y = p[1] / pixels
    x = x * 360.0 / n - 180
    y = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
    y = y * 180 / math.pi
    return (x, y)

db.execute("""
    SELECT im.time, l.column, l.row
    FROM labels AS l, windows AS w, images AS im
    WHERE im.id = w.image_id AND w.id = l.window_id
    AND (l.properties is null or l.properties not like '%orange%')
""")

with open("out.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["timestamp", "longitude", "latitude"])
    writer.writeheader()

    for ts_str, col, row in db.fetchall():
        lon, lat = mercator_to_geo((col, row), zoom=13, pixels=512)
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S%z")
        writer.writerow({
            "timestamp": ts.isoformat(),
            "longitude": str(lon),
            "latitude": str(lat),
        })
