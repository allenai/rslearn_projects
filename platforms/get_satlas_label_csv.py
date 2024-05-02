"""
Copied from vessels/get_label_csv.py.
But this one gets the platform labels instead.
Mainly intended for getting power line vs wind turbine vs platform to Viraj.
It operates on the siv_l1c/siv.sqlite3 on AWS VM.
"""
import csv
from datetime import datetime
import math

from siv.db import db

categories = ["turbine","platform","vessel","rock","power","aerialway"]

def mercator_to_geo(p, zoom, pixels):
    n = 2**zoom
    x = p[0] / pixels
    y = p[1] / pixels
    x = x * 360.0 / n - 180
    y = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
    y = y * 180 / math.pi
    return (x, y)

db.execute("""
    SELECT l.column, l.row, l.category_id
    FROM labels AS l, windows AS w
    WHERE w.id = l.window_id AND w.dataset_id = 4 AND l.category_id IS NOT NULL AND l.overlay IS NULL
""")

with open("out.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["longitude", "latitude", "category"])
    writer.writeheader()

    for col, row, category_id in db.fetchall():
        lon, lat = mercator_to_geo((col, row), zoom=13, pixels=512)
        category = categories[int(category_id)]
        writer.writerow({
            "longitude": str(lon),
            "latitude": str(lat),
            "category": category,
        })
