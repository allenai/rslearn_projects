import os
import sys

version = sys.argv[1]
ckpt_dir = "/weka/dfive-default/rslearn-eai/projects/2025_08_27_helios_cmp/"
ckpt = "checkpoints/last.ckpt"
task_to_directory = {
    "v2_nandi_crop_type": "nandi_crop_type",
    "v2_worldcereal_cropland": "worldcereal_cropland",
    "v2_satlas_marine_infra_128": "marine_infra",
    "v2_satlas_wind_turbine_128": "wind_turbine",
    "v2_sentinel1_vessels_128": "vessel_sentinel1",
    "v2_sentinel2_vessels_128": "vessel_sentinel2",
    "v2_pastis": "pastis",
    "v2_satlas_solar_farm_128": "solar_farm",
    "vessel_classification": "landsat_vessel_classify",
    "vessel_detection": "landsat_vessel_detect"
}
tasks = {v: k for k, v in task_to_directory.items()}

for d in os.listdir(ckpt_dir):
    if d.endswith("_v1") and version == "v1":
        old_or_new = "old"
    elif d.endswith("_v2") and version == "v2":
        old_or_new = "new"
    else:
        continue

    ckpt_path = os.path.join(ckpt_dir, d, ckpt)
    task = tasks[d.replace("_" + version, "")]
    cmd = f"python do_eval_sft.py {ckpt_path} {task} {old_or_new}"
    print(cmd)
    os.system(cmd)
    print()
