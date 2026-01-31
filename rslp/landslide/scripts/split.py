import random
import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath

ds_path = UPath("/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/1k_positives")
dataset = Dataset(ds_path)
windows = dataset.load_windows(show_progress=True, workers=32)

# Randomly sample 100 windows for validation
random.seed(42)  # For reproducibility
val_windows = set(random.sample(windows, 100))

for window in tqdm.tqdm(windows):
    split = "val" if window in val_windows else "train"
    
    if "split" in window.options and window.options["split"] == split:
        continue
    window.options["split"] = split
    print("setting window.options split to:", split)
    window.save()


exit()

import hashlib
import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath

ds_path = UPath("/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/1k_positives")
dataset = Dataset(ds_path)
windows = dataset.load_windows(show_progress=True, workers=32)
for window in tqdm.tqdm(windows):
    if hashlib.sha256(window.name.encode()).hexdigest()[0] in ["0", "1"]:
        split = "val"
    else:
        split = "train"
    if "split" in window.options and window.options["split"] == split:
        continue
    window.options["split"] = split
    window.save()
