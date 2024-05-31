"""
This is a quick script to visualize the outputs from create_windows.py
(after running prepare/ingest/materialize in rslearn to get the images).

Just visualizing the ground truth data.
- Heading, length in the image.
- Ship type in the filename.
"""

import base64
import io
import json
import math
import os

import numpy as np
from PIL import Image
import tqdm

mask_size = 256
pixel_size = 2.5

example_ids = os.listdir(".")
html = "<html><body><table>"
html += "<tr><th>Image</th><th>Event ID</th><th>Type</th><th>Speed</th><th>Length</th><th>Width</th></tr>"

for example_id in tqdm.tqdm(example_ids):
    im_fname = os.path.join(example_id, "layers", "sentinel2", "R_G_B", "image.png")
    if not os.path.exists(im_fname):
        continue
    im = np.array(Image.open(im_fname))

    factor = mask_size // im.shape[0]
    im = im.repeat(axis=0, repeats=factor).repeat(axis=1, repeats=factor)

    with open(os.path.join(example_id, "info.json")) as f:
        info = json.load(f)

    # Create vector from length and cog angle.
    if info["cog"] and info["length"] and info["width"]:
        radians = math.radians(info["cog"] - 90)
        length_vector = (
            int(math.cos(radians) * info["length"] / pixel_size / 2),
            int(math.sin(radians) * info["length"] / pixel_size / 2),
        )
        ortho_radians = radians + math.pi/2
        width_vector = (
            int(math.cos(ortho_radians) * info["width"] / pixel_size / 2),
            int(math.sin(ortho_radians) * info["width"] / pixel_size / 2),
        )

        #mask = np.zeros((mask_size, mask_size, 3), dtype=np.uint8)
        center = mask_size // 2
        front = (
            center + length_vector[0],
            center + length_vector[1],
        )
        back = (
            center - length_vector[0],
            center - length_vector[1],
        )
        im[front[1]-2:front[1]+2, front[0]-2:front[0]+2, :] = [255, 0, 0]
        im[back[1]-2:back[1]+2, back[0]-2:back[0]+2, :] = [255, 255, 0]
        im[center-2:center+2, center-2:center+2, :] = [255, 255, 255]

        left = (
            center + width_vector[0],
            center + width_vector[1],
        )
        right = (
            center - width_vector[0],
            center - width_vector[1],
        )
        im[left[1]-2:left[1]+2, left[0]-2:left[0]+2, :] = [255, 255, 255]
        im[right[1]-2:right[1]+2, right[0]-2:right[0]+2, :] = [255, 255, 255]

    buf = io.BytesIO()
    Image.fromarray(im).save(buf, "PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    html += '<tr>'
    html += '<td><img src="{}" /></td>'.format(data_url)
    html += '<td>{}</td>'.format(info["event_id"])
    html += '<td>{}</td>'.format(info["type"])
    html += '<td>{}</td>'.format(info["sog"])
    html += '<td>{}</td>'.format(info["length"])
    html += '<td>{}</td>'.format(info["width"])
    html += '</tr>'

html += "</table></body></html>"
with open("index.html", "w") as f:
    f.write(html)
