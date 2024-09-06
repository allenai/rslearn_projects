"""Docker entrypoint for serving models."""

import argparse
import json
import multiprocessing
import os
import shutil
from datetime import datetime

from upath import UPath

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    import rslearn.main

    from rslp.lightning_cli import CustomLightningCLI

    rslearn.main.RslearnLightningCLI = CustomLightningCLI

    parser = argparse.ArgumentParser(description="docker_entrypoint.py")
    parser.add_argument("--ds_cfg_fname", help="Dataset configuration filename")
    parser.add_argument("--model_cfg_fname", help="Model configuration filename")
    parser.add_argument("--projection", help="JSON-encoded window projection")
    parser.add_argument("--bounds", help="JSON-encoded window bounds")
    parser.add_argument("--start_time", help="Window start time")
    parser.add_argument("--end_time", help="Window end time")
    parser.add_argument(
        "--workers", type=int, default=16, help="Workers to use in serving"
    )
    parser.add_argument("--out_path", help="Where to write the output data")
    args = parser.parse_args()

    from rslearn.serve.request import Request, serve
    from rslearn.utils import Projection

    scratch_dir = "/scratch/"
    os.makedirs(scratch_dir, exist_ok=True)
    projection = Projection.deserialize(json.loads(args.projection))
    bounds = json.loads(args.bounds)
    start_time = datetime.fromisoformat(args.start_time)
    end_time = datetime.fromisoformat(args.end_time)
    request = Request(
        projection=projection,
        bounds=bounds,
        time_range=(start_time, end_time),
    )
    serve(
        ds_cfg_fname=args.ds_cfg_fname,
        model_cfg_fname=args.model_cfg_fname,
        request=request,
        scratch_dir=scratch_dir,
        workers=args.workers,
    )

    # Upload the output layer to GCS.
    layer_dir = os.path.join(scratch_dir, "windows/default/window/layers/output")
    candidates = [
        os.path.join(layer_dir, "data.geojson"),
        os.path.join(layer_dir, "output", "geotiff.tif"),
    ]
    local_out_fname = None
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        local_out_fname = candidate
    with open(local_out_fname, "rb") as src:
        with UPath(args.out_path).open("wb") as dst:
            shutil.copyfileobj(src, dst)
