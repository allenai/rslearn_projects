#!/usr/bin/env python3
"""Script to run experiments with different dataset percentages using temporary configs."""

import os
import tempfile
import yaml
import subprocess
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


def create_temp_config(base_config_paths, limit_train_batches, temp_dir, substitutions=None):
    """Create a temporary config with limit_train_batches set and adjusted epochs."""
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
    
    # Create temporary config file
    base_name = Path(base_config_path).stem
    temp_config_path = os.path.join(temp_dir, f"{base_name}_limit_{limit_train_batches}.yaml")
    save_yaml_config(config, temp_config_path)
    
    temp_config_paths = [temp_config_path]
    if isinstance(base_config_paths, list):
        temp_config_paths.extend(base_config_paths[1:])

    return temp_config_paths


def run_experiment(config_paths, experiment_name, limit_train_batches, workspace_name, args):
    """Run a single experiment."""
    env = os.environ.copy()
    env["RSLP_PREFIX"] = "/weka/dfive-default/rslearn-eai"
    cmd = [
        "python", "-m", "rslp.main", "helios", "launch_finetune",
        "--helios_checkpoint_path", args.helios_checkpoint_path,
        "--patch_size", str(args.patch_size),
        "--encoder_embedding_size", str(args.encoder_embedding_size),
        "--image_name", args.image_name,
        "--cluster+=ai2/titan-cirrascale",
        "--cluster+=ai2/ceres-cirrascale", 
        "--cluster+=ai2/saturn-cirrascale",
        "--rslp_project", workspace_name,
        "--experiment_id", f"{experiment_name}_train_{limit_train_batches}"
    ]
    for config_path in config_paths:
        cmd.append(f"--config_paths+={config_path}")
    
    print(f"Running experiment: {experiment_name} with limit_train_batches={limit_train_batches}")
    print(f"Command: {' '.join(cmd)}")
    
    # Check if we should actually run the command
    if args.run_commands:
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
    parser.add_argument("--workspace_name", type=str, default="2025_07_23_finetune_dataset_percents", 
                       help="Name of the workspace/project")
    parser.add_argument("--run_commands", action="store_true", 
                       help="Actually run the commands (default is to just print them)")
    parser.add_argument("--helios_checkpoint_path", type=str, 
                       default="/weka/dfive-default/ryanp/rslearn_projects/project_data/projects/helios_finetune_cosine_lr/soup/checkpoints/",
                       help="Path to helios checkpoint")
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
    
    # Define experiments
    experiments = [
        {
            "name": "landsat_vessel_detect",
            "config_path": configs_dir / "v2_landsat_vessels" / "finetune_detector_cosinelr.yaml",
        },
        {
            "name": "pastis",
            "config_path": [
                configs_dir / "v2_pastis" / "basecfg_cosinelr.yaml", 
                configs_dir / "v2_pastis" / "basecfg_helios_mm.yaml"
            ]
        },
        #{
        #    "name": "worldcereal_cropland",
        #    "config_path": configs_dir / "v2_worldcereal_cropland" / "finetune_s1_s2_cosinelr.yaml",
        #}
    ]
    
    # Dataset percentages to test
    limit_train_batches_values = [1.0, 0.5, 0.1, 0.01]
    
    # Create substitutions dictionary
    substitutions = {
        "CHECKPOINT_PATH": args.helios_checkpoint_path,
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
        
        for experiment in experiments:
            print(f"\n{'='*60}")
            print(f"Processing experiment: {experiment['name']}")
            print(f"{'='*60}")
            
            # Create temporary config with limit_train_batches
            for limit_val in limit_train_batches_values:
                temp_config_paths = create_temp_config(
                    experiment['config_path'], 
                    limit_val, 
                    temp_dir,
                    substitutions
                )
                
                # Run the experiment
                run_experiment(
                    temp_config_paths,
                    experiment['name'],
                    limit_val,
                    args.workspace_name,
                    args
                )
                
                # Clean up temp config
                os.remove(temp_config_paths[0])
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Workspace: {args.workspace_name}")
        print(f"Total experiments: {len(experiments) * len(limit_train_batches_values)}")
        print(f"Dataset percentages: {limit_train_batches_values}")
        print(f"Experiments:")
        for exp in experiments:
            print(f"  - {exp['name']}")
        print(f"{'='*60}")
        if not args.run_commands:
            print("To actually run the experiments, use: --run_commands")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
