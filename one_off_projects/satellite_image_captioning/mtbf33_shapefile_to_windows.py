"""Create GeoJSON containing all buildings from the MTBF-33 dataset that were constructed
in 2014 or later.
"""

import json
import multiprocessing
import os

import fiona
import tqdm

in_dir = "/data/favyenb/datasets_for_image_caption/mtbf33/"
out_fname = "mtbf33_after_2014.geojson"


def process(fname):
    features = []
    with fiona.open(os.path.join(in_dir, fname)) as src:
        for feat in src:
            if feat.properties["year_built"] < 2014:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": dict(feat.geometry),
                    "properties": dict(feat.properties),
                }
            )
    return features


fnames = [fname for fname in os.listdir(in_dir) if fname.endswith(".shp")]
fc = {
    "type": "FeatureCollection",
    "features": [],
}
p = multiprocessing.Pool(32)
outputs = p.imap_unordered(process, fnames)
for features in tqdm.tqdm(outputs, total=len(fnames)):
    fc["features"].extend(features)

with open(out_fname, "w") as f:
    json.dump(fc, f)
