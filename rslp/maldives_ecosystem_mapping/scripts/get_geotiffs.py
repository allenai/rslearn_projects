"""Download all or a subset of image and output GeoTIFFs."""

import argparse
import os
import random
import shutil

from upath import UPath

PATHS = {
    "maxar": "windows/images/{maxar_window_name}/layers/maxar/R_G_B/geotiff.tif",
    "maxar_output": "windows/images/{maxar_window_name}/layers/output/output/geotiff.tif",
    "planetscope": "windows/images_planetscope/{other_window_name}_planetscope/layers/planetscope/b01_b02_b03_b04_b05_b06_b07_b08/geotiff.tif",
    "planetscope_output": "windows/images_planetscope/{other_window_name}_planetscope/layers/output/output/geotiff.tif",
    "sentinel2": "windows/images_sentinel2/{other_window_name}_sentinel2/layers/sentinel2/B02_B03_B04_B08/geotiff.tif",
    "sentinel2_output": "windows/images_sentinel2/{other_window_name}_sentinel2/layers/output/output/geotiff.tif",
    "skysat": "windows/images_skysat/{other_window_name}_skysat/layers/planet/b01_b02_b03_b04/geotiff.tif",
    "skysat_vis": "windows/images_skysat/{other_window_name}_skysat/layers/skysat_vis/b03_b02_b01/geotiff.tif",
    "mapbox": "windows/images_skysat/{other_window_name}_skysat/layers/mapbox/R_G_B/geotiff.tif",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download GeoTIFF images",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Path to rslearn dataset for Maldives ecosystem mapping project",
        default="gs://rslearn-eai/datasets/maldives_ecosystem_mapping/dataset_v1/20240924/",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Local directory to write the GeoTIFFs",
        required=True,
    )
    parser.add_argument(
        "--count",
        type=int,
        help="How many GeoTIFFs to download (default is to get all of them)",
        default=None,
    )
    parser.add_argument(
        "--types",
        type=str,
        help="Comma-separated list of image types to download: maxar, maxar_output, planetscope, planetscope_output, sentinel2, sentinel2_output, skysat, skysat_vis, mapbox",
        required=True,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    wanted = args.types.split(",")

    maxar_window_names = [
        path.name for path in (ds_path / "windows" / "images").iterdir()
    ]
    other_window_names = [
        path.name.split("_planetscope")[0]
        for path in (ds_path / "windows" / "images_planetscope").iterdir()
    ]

    # Determine which list of windows to use as the basis for sampling.
    # Maxar is limited to ~100 windows while the other data is available more broadly.
    # So we limit to Maxar if maxar/maxar_output is requested.
    if "maxar" in wanted or "maxar_output" in wanted:
        # OK so we need to match between the Maxar and other window names then.
        window_names = []
        for name1 in maxar_window_names:
            island1 = name1.split("_")[0]
            for name2 in other_window_names:
                island2 = name2.split("_")[1].lower().replace("'", "")
                if island1 == island2:
                    window_names.append(
                        dict(
                            maxar_window_name=name1,
                            other_window_name=name2,
                        )
                    )
                    break

    else:
        window_names = [dict(other_window_name=name) for name in other_window_names]

    if args.count:
        window_names = random.sample(window_names, args.count)

    for window_name in window_names:
        print(window_name)
        for t in wanted:
            path_tmpl = PATHS[t]
            actual_path = ds_path / path_tmpl.format(**window_name)
            if not actual_path.exists():
                continue
            local_fname = os.path.join(
                args.out_dir, window_name["other_window_name"] + "_" + t + ".tif"
            )
            with actual_path.open("rb") as src:
                with open(local_fname, "wb") as dst:
                    shutil.copyfileobj(src, dst)
