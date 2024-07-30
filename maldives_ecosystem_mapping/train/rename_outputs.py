"""
After using `python rslp.main model predict`, this renames the outputs from the layers
directory to a flat directory.
"""

import glob
import os

fnames = glob.glob('*/layers/output/output/geotiff.tif')
for fname in fnames:
    os.rename(fname, os.path.join("outputs", fname.split("/")[0]+".tif"))
