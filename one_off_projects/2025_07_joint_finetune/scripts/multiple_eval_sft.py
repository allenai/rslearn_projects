import os

old_ckpt_home_dir = "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/"
new_ckpt_home_dir = "/weka/dfive-default/rslearn-eai/projects/2025_07_29_helios_finetune/"
ckpt = "checkpoints/last.ckpt"
jobs = {
    "v2_sentinel1_vessels_128": os.path.join(old_ckpt_home_dir, "vessel_sentinel1", ckpt),
    "v2_sentinel2_vessels_128": os.path.join(old_ckpt_home_dir, "vessel_sentinel2", ckpt),
    "v2_satlas_marine_infra_128": os.path.join(old_ckpt_home_dir, "marine_infra", ckpt),
    "v2_satlas_wind_turbine_128" : os.path.join(old_ckpt_home_dir, "wind_turbine", ckpt),
    "vessel_detection": os.path.join(new_ckpt_home_dir, "landsat_vessel_detect", ckpt)
}

for task, ckpt_path in jobs.items():
    os.system(f"python do_eval_sft.py {ckpt_path} {task}")
