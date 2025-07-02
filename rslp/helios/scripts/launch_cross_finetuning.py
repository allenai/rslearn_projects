import os
import subprocess
import concurrent.futures

cmd1 = """
RSLP_PREFIX=gs://rslearn-eai \
    python -m rslp.main helios launch_finetune \
        --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/ryanp/{checkpoint} \
        --patch_size {patch_size} \
        --encoder_embedding_size {embedding_size} \
        --image_name {image_name} \
        --config_paths+=data/helios/{benchmark}/basecfg.yaml \
        --config_paths+=data/helios/{benchmark}/basecfg_helios_mm.yaml \
        --config_paths+=data/helios/v2_shared/helios_freeze_then_lowlr.yaml \
        --cluster+=ai2/ceres-cirrascale \
        --cluster+=ai2/saturn-cirrascale \
        --rslp_project {rslp_project} \
        --experiment_id {experiment_id} \
        --priority {priority}
"""

cmd2 = """
RSLP_PREFIX=gs://rslearn-eai \
    python -m rslp.main helios launch_finetune \
        --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/ryanp/{checkpoint} \
        --patch_size {patch_size} \
        --encoder_embedding_size {embedding_size} \
        --image_name {image_name} \
        --config_paths+=data/helios/{benchmark}/{cfg}.yaml \
        --cluster+=ai2/ceres-cirrascale \
        --cluster+=ai2/saturn-cirrascale \
        --rslp_project {rslp_project} \
        --experiment_id {experiment_id} \
        --priority {priority}
"""

char_limit = 90  # beaker strings can't be >128 chars
homepath = "/weka/dfive-default/ryanp/rslearn_projects"
priority = "normal"
rslp_project = "helios_cross_finetuning"
patch_size = 8
embedding_size = 768
image_name = "favyen/rslphelios3"
base_path = "/weka/dfive-default/ryanp/rslearn_projects/data/helios"
models = [
    "v2_crop_type_classification_helios_base_S2_ts_ws8_ps1",
    "v2_landsat_vessel_classification_helios_base_ps4_add_prob_threshold",
    "v2_landsat_vessel_detection_helios_base_ps4",
    "v2_cropland_classification_helios_base_S2_ts_ws8_ps8",
    "v2_base"
]
benchmarks = {
    "v2_pastis": None,
    "v2_nandi_crop_type": ["finetune_s2"],
    "v2_worldcereal_cropland": ["finetune_s2"],
    "v2_landsat_vessels": ["finetune_classifier", "finetune_detector"],
}

commands = []
for benchmark, info in benchmarks.items():
    for model in models:
        if not model.startswith("v2_") and not model.startswith("base"):
            continue
        if info is not None:
            cmd_template = cmd2
            for cfg in info:
                cmd = cmd_template.format(
                    checkpoint=model,
                    patch_size=patch_size,
                    embedding_size=embedding_size,
                    image_name=image_name,
                    benchmark=benchmark,
                    rslp_project=rslp_project,
                    experiment_id=f"{benchmark}_{cfg}__CROSS__{model}"[:char_limit],
                    priority=priority,
                    cfg=cfg,
                )
                commands.append(cmd.strip())
        else:
            cmd_template = cmd1
            cmd = cmd_template.format(
                checkpoint=model,
                patch_size=patch_size,
                embedding_size=embedding_size,
                image_name=image_name,
                benchmark=benchmark,
                rslp_project=rslp_project,
                experiment_id=f"{benchmark}__CROSS__{model}"[:char_limit],
                priority=priority,
            )
            commands.append(cmd.strip())

print("=" * 80)
for cmd in commands:
    print(cmd)
    print("\n\n")
print("=" * 80)

os.chdir(homepath)

print("Number of models:", len(models))
print("Number of benchmarks:", len(benchmarks))
print("Number of commands:", len(commands))
print("=" * 80)
input("Press Enter to continue...")

def run_command(cmd):
    print(f"Running: {cmd}\n\n")
    try:
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

max_workers = min(32, len(commands))
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(run_command, commands))

successful = sum(results)
failed = len(results) - successful
print(f"\nSummary: {successful} successful, {failed} failed")
