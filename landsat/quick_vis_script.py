"""
Visualizing prediction outputs from sentinel-vessel-detection example.
"""

import csv
import numpy as np
from PIL import Image
import rasterio
prediction_fname = "predictions_new.csv"
out_fname = "new.png"
im = rasterio.open("/home/favyen/big-documents/research/allenai/sentinel-vessel-detection/example/data/landsat/LC09_L1TP_162042_20240417_20240417_02_T1/LC09_L1TP_162042_20240417_20240417_02_T1_B8.TIF").read(1)
im = np.clip((im-4000)//50, 0, 255).astype(np.uint8)
with open(prediction_fname) as f:
    reader = csv.DictReader(f)
    for csv_row in reader:
        col = int(csv_row["column"])
        row = int(csv_row["row"])
        im[row-10:row+10, col-10:col-8] = 255
        im[row-10:row+10, col+8:col+10] = 255
        im[row-10:row-8, col-10:col+10] = 255
        im[row+8:row+10, col-10:col+10] = 255
Image.fromarray(im).save(out_fname)
