import os

ckpt_dir = "/weka/dfive-default/helios/checkpoints/ryanp/"
config_dir = "/weka/dfive-default/ryanp/rslearn_projects/data/helios"
root_dir = "/weka/dfive-default/ryanp/rslearn_projects"
token = "__CROSS__"
patch_size = 8
encoder_embedding_size = 768
image_name = "favyen/rslphelios3"
tasks = [
    ("v2_landsat_vessels", "finetune_classifier"),
    ("v2_landsat_vessels", "finetune_detector"),
    ("v2_nandi_crop_type", "finetune_s2"),
    ("v2_worldcereal_cropland", "finetune_s2"),
]
task_name_map = {
    "v2_landsat_vessel_classification": tasks[0],
    "v2_landsat_vessel_detection": tasks[1],
    "v2_crop_type_classification": tasks[2],
    "v2_cropland_classification": tasks[3],
}
cmd = """
python -m rslp.main helios launch_finetune \
    --helios_checkpoint_path {helios_checkpoint_path} \
    --patch_size {patch_size} \
    --encoder_embedding_size {encoder_embedding_size} \
    --image_name {image_name} \
    --cluster+=ai2/ceres-cirrascale \
    --config_paths+={config_path} \
    --experiment_id {run}__{task}__{cfg} \
    --rslp_project helios_evals \
    --local true \
    --do_eval true \
"""

def find_task_and_cfg(task_and_cfg):
    for task in tasks:
        if task in task_and_cfg:
            cfg = task_and_cfg.replace(f"{task}_", "")
            return task, cfg
    raise ValueError(f"No matching task found for {task_and_cfg}")

def get_base_task(run):
    """Get base finetuning task from run name"""
    s = run.split(token)[1]
    search_str = "_helios_"
    if search_str not in s:
        return run
    return s[:s.find(search_str)]


cmds = []
for run in os.listdir(ckpt_dir):
    if token in run:
        ckpt_path = os.path.join(ckpt_dir, run)
        for task, cfg in tasks:
            task_matches_base = (task_name_map.get(get_base_task(run)) == (task, cfg))
            task_matches_finetune = (f"{task}_{cfg}" == run.split(token)[0])
            if task_matches_base or task_matches_finetune:
                config_path = os.path.join(config_dir, task, cfg + ".yaml")
                filled_cmd = cmd.format(
                    helios_checkpoint_path=ckpt_path,
                    patch_size=patch_size,
                    encoder_embedding_size=encoder_embedding_size,
                    image_name=image_name,
                    config_path=config_path,
                    run=run,
                    task=task,
                    cfg=cfg
                )
                print(filled_cmd)
                print()
                cmds.append(filled_cmd)

print(len(cmds))
input("Continue? ")

os.chdir(root_dir)
for cmd in cmds:
    os.system(cmd)
