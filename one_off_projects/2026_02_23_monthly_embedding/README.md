This was to verify that we can use rslearn to compute per-timestep embeddings with
OlmoEarth. So we input a calendar year of data and get the embedding for each month
(concatenated with the 768 embedding dimensions on the channel axis). The modalities
get mixed in there too so we still need to do some pooling in post-processing.

Create the dataset:

```
export DATASET_PATH=./dataset
mkdir $DATASET_PATH
cp one_off_projects/2026_02_23_monthly_embedding/config.json $DATASET_PATH/config.json
rslearn dataset add_windows --root $DATASET_PATH --group default --utm --resolution 10 --window_size 1024 --src_crs EPSG:4326 --box=-122.255,47.589,-122.255,47.589 --start 2025-01-01T00:00:00+00:00 --end 2026-01-01T00:00:00+00:00 --name seattle
```

Materialize the data:

```
rslearn dataset prepare --root $DATASET_PATH --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root $DATASET_PATH --retry-max-attempts 5 --retry-backoff-seconds 5
```

Apply the model and get embeddings:

```
rslearn model predict --config one_off_projects/2026_02_23_monthly_embedding/model.yaml --data.init_args.path=$DATASET_PATH
```

Use this script to pool within each timestep and get one GeoTIFF per timestep:

```python
import os
from pathlib import Path

import rasterio
from einops import rearrange

num_bandsets = 3
num_timesteps = 12
embedding_dim = 768

ds_path = Path(os.environ["DATASET_PATH"])
print("Read embeddings")
in_fname = next((ds_path / "windows" / "default" / "seattle" / "layers" / "embeddings").glob("*/geotiff.tif"))
with rasterio.open(in_fname) as raster:
    array = raster.read()
    profile = raster.profile.copy()

print("Get per-timestep embeddings")
per_timestep = rearrange(
    array, "(t s c) h w -> t c h w s", t=num_timesteps, s=num_bandsets, c=embedding_dim
).mean(axis=4)

for timestep_idx, timestep_embeddings in enumerate(per_timestep):
    out_fname = in_fname.parent / f"geotiff_{timestep_idx}.tif"
    print(f"Write to {out_fname}")
    profile.update(count=embedding_dim, dtype=timestep_embeddings.dtype)
    with rasterio.open(out_fname, "w", **profile) as dst:
        dst.write(timestep_embeddings)
```