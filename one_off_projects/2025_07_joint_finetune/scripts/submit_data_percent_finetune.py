#!/usr/bin/env python3
"""Script to run experiments with different dataset percentages using temporary configs."""

import os
import tempfile
import yaml
import subprocess
import json
from pathlib import Path


def load_yaml_config(config_path, substitutions=None):
    """Load a YAML config file with optional string substitutions."""
    with open(config_path, 'r') as f:
        config_str = f.read()
    
    # Apply substitutions if provided
    if substitutions:
        for key, value in substitutions.items():
            if value is not None:
                config_str = config_str.replace(f"{{{key}}}", str(value))

    return yaml.safe_load(config_str)


def save_yaml_config(config, config_path):
    """Save a YAML config to file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_temp_config(
    base_config_paths,
    limit_train_batches,
    temp_dir,
    substitutions=None,
    model_data=None
):
    """Create a temporary config with limit_train_batches set and adjusted epochs."""
    if model_data is None:
        model_data = {}
    if isinstance(base_config_paths, list):
        base_config_path = base_config_paths[0]
    else:
        base_config_path = base_config_paths

    config = load_yaml_config(base_config_path, substitutions)
    
    # Add limit_train_batches to trainer section
    if 'trainer' not in config:
        config['trainer'] = {}
    config['trainer']['limit_train_batches'] = limit_train_batches
    
    # Adjust max_epochs inversely to maintain same number of gradient updates
    if 'max_epochs' in config['trainer']:
        original_epochs = config['trainer']['max_epochs']
        adjusted_epochs = int(original_epochs / limit_train_batches)
        config['trainer']['max_epochs'] = adjusted_epochs
        print(f"  Adjusted epochs: {original_epochs} -> {adjusted_epochs} (factor: {1/limit_train_batches:.2f})")

    # Add restore config
    if model_data.get("sft") is not None:
        config['model']['init_args']['restore_config'] = {
            "restore_path": os.path.join(model_data["sft"], "checkpoints", "last.ckpt"),
            "selector": ["state_dict"],
            "remap_prefixes": [["model.", ""]]
        }
        config['model']['init_args']['model']['init_args']['task_embedding'] = {
            "class_path": "rslearn.models.task_embedding.TaskMHAEmbedding",
            "init_args": {
                "encoder_embedding_size": 768,
                "num_heads": 12,
            }
        }
        print(f"  Restoring from: {model_data['sft']}/checkpoints/last.ckpt")
        print(f"  Task embedding: {json.dumps(config['model']['init_args']['model']['init_args']['task_embedding'], indent=4)}")

    # Create temporary config file
    base_name = Path(base_config_path).stem
    temp_config_path = os.path.join(temp_dir, f"{base_name}_limit_{limit_train_batches}.yaml")
    save_yaml_config(config, temp_config_path)
    
    temp_config_paths = [temp_config_path]
    if isinstance(base_config_paths, list):
        temp_config_paths.extend(base_config_paths[1:])

    return temp_config_paths


def run_experiment(
    config_paths,
    experiment_name,
    limit_train_batches,
    model_data,
    model_name,
    workspace_name,
    run_commands,
    patch_size,
    encoder_embedding_size,
    image_name,
):
    """Run a single experiment."""
    env = os.environ.copy()
    env["RSLP_PREFIX"] = "/weka/dfive-default/rslearn-eai"
    cmd = [
        "python", "-m", "rslp.main", "helios", "launch_finetune",
        "--helios_checkpoint_path", model_data["helios"],
        "--patch_size", str(patch_size),
        "--encoder_embedding_size", str(encoder_embedding_size),
        "--image_name", image_name,
        "--cluster+=ai2/titan-cirrascale",
        "--cluster+=ai2/ceres-cirrascale", 
        "--cluster+=ai2/saturn-cirrascale",
        "--rslp_project", workspace_name,
        "--experiment_id", f"{model_name}__{experiment_name}__{limit_train_batches}"
    ]
    for config_path in config_paths:
        cmd.append(f"--config_paths+={config_path}")
    
    print(f"Running experiment: {experiment_name} with limit_train_batches={limit_train_batches} @ {model_name}")
    print(f"Command: {' '.join(cmd)}")
    
    # Check if we should actually run the command
    if run_commands:
        print("Executing command...")
        subprocess.run(cmd, check=True, env=env)
    else:
        print("Command would be executed (use --run_commands to actually run)")
    print()


def main():
    """Main function to run all experiments."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiments with different dataset percentages")
    parser.add_argument("--workspace_name", type=str, default="2025_07_30_stage1_fewshot", 
                       help="Name of the workspace/project")
    parser.add_argument("--run_commands", action="store_true", 
                       help="Actually run the commands (default is to just print them)")
    parser.add_argument("--patch_size", type=int, default=8,
                       help="Patch size for the model")
    parser.add_argument("--encoder_embedding_size", type=int, default=768,
                       help="Encoder embedding size")
    parser.add_argument("--image_name", type=str, default="henryh/rslp_multidataset_dev",
                       help="Docker image name")
    args = parser.parse_args()
    
    # Configuration
    base_dir = Path(__file__).parent.parent
    configs_dir = base_dir / "configs"
    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")

    # Define checkpoint paths
    ckpt_paths = {
        "v2_base": {
            "helios": "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
            "sft": None,
        },
        # "classify_v2": {
        #     "helios": "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
        #     "sft": "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/classify_all_v2__unmerged__vessel_classification"
        # },
        # "detect_v2": {
        #     "helios": "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
        #     "sft": "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/detect_all_v2__unmerged__vessel_detection"
        # },
        # "segment_v2": {
        #     "helios": "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
        #     "sft": "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/segment_all_v2__unmerged__segment"
        # },
    }
    
    # Define experiments
    experiments = [
        #{
        #    "name": "vessel_detection",
        #    "config_paths": [
        #        configs_dir / "v2_landsat_vessels" / "finetune_detector_cosinelr.yaml",
        #        #"/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml"
        #        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v3_singletask/freeze.yaml"
        #    ],
        #},
        #{
        #    "name": "cropland_classification",
        #    "config_paths": [
        #        configs_dir / "v2_worldcereal_cropland" / "finetune_s1_s2_cosinelr.yaml",
        #        "/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml"
        #    ],
        #},
        #{
        #    "name": "nandi_crop_type",
        #    "config_paths": [
        #        configs_dir / "v2_nandi_crop_type" / "finetune_s1_s2_cosinelr.yaml",
        #        #"/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml"
        #        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v3_singletask/freeze.yaml"
        #    ],
        #},
        {
            "name": "marine_infra",
            "config_paths": [
                configs_dir / "v2_satlas_marine_infra_128" / "basecfg_cosinelr.yaml",
                configs_dir / "v2_satlas_marine_infra_128" / "basecfg_helios_mm.yaml",
                "/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml",
            ]
        },
        #{
        #    "name": "sentinel1_vessels",
        #    "config_paths": [
        #        configs_dir / "v2_sentinel1_vessels_128" / "basecfg_cosinelr.yaml",
        #        configs_dir / "v2_sentinel1_vessels_128" / "basecfg_helios.yaml",
        #        "/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml",
        #    ]
        #},
    ]
    
    # Dataset percentages to test
    limit_train_batches_values = [0.01, 0.1, 0.2, 0.5, 0.7, 0.9]
    
    # Create substitutions dictionary
    substitutions = {
        "PATCH_SIZE": args.patch_size,
        "ENCODER_EMBEDDING_SIZE": args.encoder_embedding_size,
        "256/PATCH_SIZE": 256 // args.patch_size,
        "128/PATCH_SIZE": 128 // args.patch_size,
    }
    
    # Create temporary directory for configs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        print(f"Workspace name: {args.workspace_name}")
        print(f"Run mode: {'EXECUTE' if args.run_commands else 'DRY RUN'}")
        print(f"Substitutions: {substitutions}")
        print()

        for model_name, model_data in ckpt_paths.items():

            for experiment in experiments:
                print(f"\n{'='*60}")
                print(f"Processing experiment: {experiment['name']} @ {model_name}")
                print(f"{'='*60}")
                
                # Create temporary config with limit_train_batches
                for limit_val in limit_train_batches_values:
                    substitutions["CHECKPOINT_PATH"] = model_data["helios"]
                    temp_config_paths = create_temp_config(
                        experiment['config_paths'], 
                        limit_val,
                        temp_dir,
                        substitutions,
                        model_data
                    )
 
                    # Run the experiment
                    run_experiment(
                        temp_config_paths,
                        experiment['name'],
                        limit_val,
                        model_data,
                        model_name,
                        args.workspace_name,
                        args.run_commands,
                        args.patch_size,
                        args.encoder_embedding_size,
                        args.image_name,
                    )

                    # Clean up temp config
                    os.remove(temp_config_paths[0])
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Workspace: {args.workspace_name}")
        print(f"Total experiments: {len(experiments) * len(limit_train_batches_values) * len(ckpt_paths)}")
        print(f"Dataset percentages: {limit_train_batches_values}")
        print(f"Models: {list(ckpt_paths.keys())}")
        print(f"Experiments:")
        for exp in experiments:
            print(f"  - {exp['name']}")
        print(f"{'='*60}")
        if not args.run_commands:
            print("To actually run the experiments, use: --run_commands")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
