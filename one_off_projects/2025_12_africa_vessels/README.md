This is for UPF request to get vessel detections in Africa.

First run script to get the 5x5 degree AOIs that are not on land:

```
python one_off_projects/2025_12_africa_vessels/get_aois.py
```

Then we can get scene IDs for each AOI, e.g.:

```
python one_off_projects/2025_12_africa_vessels/get_scene_ids.py \
    --cache_path /weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/cache/sentinel2/ \
    --geojson /weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/aois/aoi_0_0_5_5.geojson \
    --out_fname /weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/scene_ids/aoi_0_0_5_5.json \
    --geom_fname /weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/scene_geojsons/aoi_0_0_5_5.geojson
```

The `--out_fname` has a simple list of scene IDs compatible with
`rslp.main sentinel2_vessels write_entries`, while `--geom_fname` has the detailed
scene geometry that could be useful for UPF.

Here is batch version:

```python
import multiprocessing
import os
import subprocess

import tqdm

def process(aoi_name: str) -> None:
    subprocess.call([
        "python",
        "one_off_projects/2025_12_africa_vessels/get_scene_ids.py",
        "--cache_path=/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/cache/sentinel2/",
        f"--geojson=/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/aois/{aoi_name}.geojson",
        f"--out_fname=/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/scene_ids/{aoi_name}.json",
        f"--geom_fname=/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/scene_geojsons/{aoi_name}.geojson",
    ])

aoi_names = [fname.split(".")[0] for fname in os.listdir("/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/aois/")]
p = multiprocessing.Pool(4)
outputs = p.imap_unordered(process, aoi_names)
for _ in tqdm.tqdm(outputs, total=len(aoi_names)):
    pass
p.close()
```

Write the jobs to queue:

```
python -m rslp.main sentinel2_vessels write_entries \
    --queue_name favyen/sentinel2-vessels-predict \
    --json_fname /weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/scene_ids/aoi_0_0_5_5.json  \
    --json_out_dir /weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/json_outputs/ \
    --geojson_out_dir /weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/geojson_outputs/ \
    --crop_out_dir /weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/crop_outputs/
```

Here is batch version:

```python
import os
import subprocess
for fname in os.listdir("/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/scene_ids/"):
    label = fname.split(".")[0]
    subprocess.call([
        "python",
        "-m",
        "rslp.main",
        "sentinel2_vessels",
        "write_entries",
        "--queue_name",
        "favyen/sentinel2-vessels-predict",
        "--json_fname",
        f"/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/scene_ids/{label}.json",
        "--json_out_dir",
        "/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/json_outputs/",
        "--geojson_out_dir",
        "/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/geojson_outputs/",
        "--crop_out_dir",
        "/weka/dfive-default/rslearn-eai/projects/2025_12_africa_vessels/crop_outputs/",
    ])
```

And launch worker jobs:

```
python -m rslp.main common launch --image_name favyen/rslpomp20251212c --queue_name favyen/sentinel2-vessels-predict --num_workers 100 --gpus 1 --shared_memory 256GiB --cluster=[ai2/jupiter,ai2/neptune,ai2/saturn] --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}'
```
