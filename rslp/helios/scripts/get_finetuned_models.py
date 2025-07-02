"""
Get the best finetuned models from W&B and download them to a local directory.
Usage: WANDB_PROJECT=?? WANDB_ENTITY=?? GCLOUD_PROJECT_ID=?? python3 get_finetuned_models.py
"""

import subprocess
import os
import shutil
import wandb
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

BEAKER_WORKSPACE = "ai2/earth-systems"
PROJECT_ID = os.environ["GCLOUD_PROJECT_ID"]
PROJECT = os.environ["WANDB_PROJECT"]
ENTITY = os.environ["WANDB_ENTITY"]


def get_best_runs():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    metric_names = set()
    for run in runs:
        for metric in run.summary.keys():
            try:
                metric_k, metric_v = metric.split("/")
            except ValueError:
                continue
            if (
                metric_k.startswith("val_") and (
                    metric_k.endswith("_detection") or metric_k.endswith("_classification")
                ) and
                metric_v in ("accuracy", "F1")
            ):
                metric_names.add(metric)

    best_runs = {}
    for metric in metric_names:
        best_run = None
        best_value = None

        for run in runs:
            metric_value = run.summary.get(metric)
            if metric_value is not None:
                if best_value is None or metric_value > best_value:
                    best_value = metric_value
                    best_run = run

        if best_run:
            dirpath = best_run.config["trainer"]["callbacks"][1]["init_args"]["dirpath"]
            pretrained = best_run.config["model"]["init_args"]["model"]["init_args"]["encoder"][0]["init_args"]["checkpoint_path"]
            best_runs[metric] = {
                "run_name": best_run.name,
                "dirpath": dirpath,
                "pretrained": pretrained
            }

    return best_runs
 

if __name__ == "__main__":
    user = subprocess.check_output(
        "beaker account whoami --format json | jq -r '.[0].name'", shell=True
    ).decode("utf-8").strip()
    local_dir = f"/weka/dfive-default/helios/checkpoints/{user}/"
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading finetuned models to {local_dir}")

    best_runs = get_best_runs()
    print(json.dumps(best_runs, indent=4))

    def download_blob_async(blob_path, local_path, metric_name):
        """Download a single blob asynchronously."""
        try:
            download_cmd = ["gcloud", "storage", "cp", blob_path, local_path]
            print(f"[{metric_name}] Downloading: {' '.join(download_cmd)}")
            subprocess.run(download_cmd, check=True)
            print(f"[{metric_name}] Downloaded {blob_path} to {local_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[{metric_name}] Error downloading {blob_path}: {e}")
            return False
        except Exception as e:
            print(f"[{metric_name}] Unexpected error downloading {blob_path}: {e}")
            return False

    # Collect all download tasks
    download_tasks = []
    checkpoint = "last.ckpt"
    
    for metric, run_info in best_runs.items():
        gs_path = run_info["dirpath"]
        bucket_name = gs_path.split("/")[2]
        blob_path = os.path.join("/".join(gs_path.split("/")[3:]), checkpoint)
        
        print(f"Processing {metric}: {gs_path}")
        print(f"Looking for: gs://{bucket_name}/{blob_path}")
        
        try:
            print(f"Listing blobs with prefix: {blob_path}")
            list_cmd = ["gcloud", "storage", "ls", f"gs://{bucket_name}/{blob_path}"]
            print(f"Running: {' '.join(list_cmd)}")
            result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
            blob_paths = result.stdout.strip().split('\n')
            
            for blob_path in blob_paths:
                if blob_path and not blob_path.endswith('/'):
                    # Extract the filename from the blob path
                    blob_name = blob_path.split('/')[-1]
                    local_path = os.path.join(local_dir, run_info["run_name"], blob_name)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Add to download tasks
                    download_tasks.append((blob_path, local_path, metric))

                    # Also, copy the config.json file from "pretrained" key in run_info
                    config_dest = local_path.replace(checkpoint, "config.json")
                    config_src = os.path.join(run_info["pretrained"], "config.json")
                    print(f"Copying config.json from {config_src} to {config_dest}")
                    shutil.copy(config_src, config_dest)
 
        except subprocess.CalledProcessError as e:
            print(f"Error processing {metric}: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error processing {metric}: {e}")

    # Execute all downloads in parallel
    print(f"\nStarting parallel download of {len(download_tasks)} files...")
    max_workers = min(10, len(download_tasks))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(download_blob_async, blob_path, local_path, metric): (blob_path, local_path, metric)
            for blob_path, local_path, metric in download_tasks
        }
        
        completed = 0
        failed = 0
        for future in as_completed(future_to_task):
            blob_path, local_path, metric = future_to_task[future]
            try:
                success = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
                print(f"Progress: {completed + failed}/{len(download_tasks)} completed (success: {completed}, failed: {failed})")
            except Exception as e:
                failed += 1
                print(f"Exception in download task for {blob_path}: {e}")
    
    print(f"\nDownload complete! Success: {completed}, Failed: {failed}")
