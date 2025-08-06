#!/usr/bin/env python
"""Prepare a model trained across various instances of a task for a single dataset."""

import os
import re
import argparse
import yaml
import json
import torch
from typing import Any, Dict, List

USE_TMP_TASK_ORDER = False
_tmp_task_order = {
    # Forgot to order the decoder tasks in the original runs, using this for now
    "classify": [
        "vessel_classification",
        "crop_type_classification",
        "cropland_classification",
    ],
    "segment": [
        "segment",
        "segment_satlas_solar_farm"
    ],
    "detect": [
        "vessel_detection",
        "detect_satlas_marine_infra",
        "detect_satlas_wind_turbine",
        "detect_sentinel1_vessels",
        "detect_sentinel2_vessels"
    ]
}


def recursive_find_key(d: Dict[str, Any], target: str) -> Any:
    """Search dict d (possibly nested) for key==target and return its value.

    Args:
        d (dict): The dictionary to search.
        target (str): The key to search for.
    Returns:
        The value of the key, or None if the key is not found.
    """
    if target in d:
        return d[target]
    for v in d.values():
        if isinstance(v, dict):
            found = recursive_find_key(v, target)
            if found is not None:
                return found
    return None


def slice_sd(
    sd: dict[str, torch.Tensor],
    key: str,
    keep_idx: List[int],
    delete: bool = False,
) -> dict[str, torch.Tensor]:
    """Index the first dim of tensor by key and then key by keep_idx.

    Args:
        sd (dict[str, torch.Tensor]): The state dict to index.
        key (str): The key to index.
        keep_idx (List[int]): The indices to keep.
        delete (bool): Whether to delete the original keys.
    Returns:
        dict[str, torch.Tensor]: The indexed state dict.
    """
    x = sd[key]
    if delete:
        del sd[key]
    return x[keep_idx]


def rename_keys(sd: dict[str, torch.Tensor], special: str, task: str) -> None:
    """Rename keys in sd from special to task.

    Args:
        sd (dict[str, torch.Tensor]): The state dict to rename.
        special (str): The special string to replace.
        task (str): The task name.
    """
    for k in list(sd.keys()):
        if special in k:
            sd[k.replace(special, task)] = sd[k]
            del sd[k]


def trim_state_dict(
    sd: Dict[str, torch.Tensor],
    task: str,
    i: int,
    n: int,
    N: int,
    task_type: str,
    task_offsets: Dict[str, Any],
) -> None:
    """
    Mutates sd in-place, slicing out i:i+n from the relevant head of the given task.

    Args:
        sd (Dict[str, torch.Tensor]): The state dict to trim.
        task (str): The task name.
        i (int): Start index for trimming.
        n (int): Number of outputs to keep.
        N (int): Original total number of outputs.
        task_type (str): The task type (detect/classify/segment).
        task_offsets (Dict[str, Any]): The task offsets.
    """
    print(f"\n-> Trimming task '{task}': keep indices [{i}:{i+n}] out of {N}")

    # Slice the task embedding first
    wkey = "model.task_embedding.embed.weight"
    task_shape = sd[wkey].shape
    ordered_tasks = sorted(task_offsets.keys())
    if USE_TMP_TASK_ORDER:
        print(f"WARNING: true order is {ordered_tasks}, using {_tmp_task_order[task_type]}")
        ordered_tasks = _tmp_task_order[task_type]
    else:
        print(f"Using task order {ordered_tasks}")
    sd[wkey] = slice_sd(sd, wkey, [ordered_tasks.index(task)], delete=False)
    print(f"  - task_embedding: original {task_shape}, new shape={sd[wkey].shape}")

    # Then slice the rest of the state dict
    if task_type == "detect":
        special = "FasterRCNN"
        cls_w = next(k for k in sd if k.endswith("box_predictor.cls_score.weight"))
        cls_b = cls_w.replace(".weight", ".bias")
        reg_w = next(k for k in sd if k.endswith("box_predictor.bbox_pred.weight"))
        reg_b = reg_w.replace(".weight", ".bias")

        assert sd[cls_w].shape[0] == N, f"Expected {N} cls_score rows, got {sd[cls_w].shape[0]}"
        assert sd[reg_w].shape[0] == 4 * N, f"Expected {4*N} bbox rows, got {sd[reg_w].shape[0]}"

        keep = list(range(i, i + n))
        keep_reg: List[int] = []
        for k in keep:
            keep_reg += list(range(4*k, 4*k + 4))

        print(f"  - cls_score: original {sd[cls_w].shape}, new rows={len(keep)}")
        sd[cls_w] = slice_sd(sd, cls_w, keep)
        sd[cls_b] = slice_sd(sd, cls_b, keep)

        print(f"  - bbox_pred: original {sd[reg_w].shape}, new rows={len(keep_reg)}")
        sd[reg_w] = slice_sd(sd, reg_w, keep_reg)
        sd[reg_b] = slice_sd(sd, reg_b, keep_reg)

        rename_keys(sd, special, task)
        print(f"  - in all keys, replaced {special} with {task}")

    elif task_type == "classify":
        special = "ClassificationHead"
        wkey = next(k for k in sd if k.endswith(f"{special}.0.output_layer.weight"))
        bkey = wkey.replace(".weight", ".bias")
        assert sd[wkey].shape[0] == N, f"Expected {N} rows, got {sd[wkey].shape[0]}"

        keep = list(range(i, i + n))
        print(f"  - output_layer: original {sd[wkey].shape}, new rows={len(keep)}")
        sd[wkey] = slice_sd(sd, wkey, keep)
        sd[bkey] = slice_sd(sd, bkey, keep)

        rename_keys(sd, special, task)
        print(f"  - in all keys, replaced {special} with {task}")

    elif task_type == "segment":
        special = "SegmentationHead"
        pattern = re.compile(r".*SegmentationHead.*layers\.\d+\.\d+\.weight$")
        w_keys = [k for k in sd if pattern.match(k)]
        if not w_keys:
            raise KeyError(f"No {special} weight keys found for task '{task}'")
        w_keys = sorted(w_keys, key=lambda x: int(x.split(".")[-2]))
        wkey = w_keys[-1]
        bkey = wkey.replace(".weight", ".bias")
        assert sd[wkey].shape[0] == N, f"Expected {N} rows, got {sd[wkey].shape[0]}"

        keep = list(range(i, i + n))
        print(f"  - {wkey}: original {sd[wkey].shape}, new rows={len(keep)}")
        sd[wkey] = slice_sd(sd, wkey, keep)
        sd[bkey] = slice_sd(sd, bkey, keep)

        rename_keys(sd, special, task)
        print(f"  - in all keys, replaced {special} with {task}")

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim a multi-task checkpoint to smaller output head per task"
    )
    parser.add_argument(
        "ckpt_dir",
        help="Directory containing config.yaml and checkpoints"
    )
    parser.add_argument(
        "--ckpt_path", default="last.ckpt",
        help="Name of the checkpoint file inside ckpt_dir (default: last.ckpt)"
    )
    parser.add_argument(
        "--ckpt_out_dir", required=True,
        help=(
            "Template for output directories, e.g. '/out/dir/{task}/checkpoints', "
            "where {task} will be replaced by each task name"
        )
    )
    args = parser.parse_args()

    key = "task_label_offsets"
    cfg_path = os.path.join(args.ckpt_dir, "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    task_offsets = recursive_find_key(cfg, key)
    if task_offsets is None:
        raise KeyError(f"Could not find '{key}' in config.yaml")

    tasks = list(task_offsets.keys())
    print(f"Found tasks: {tasks}")

    info_path = (
        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects"
        "/2025_07_joint_finetune/configs/v3_multitask/all_dataset_info.yaml"
    )
    with open(info_path, "r") as f:
        task_types = {k: v["task_type"] for k, v in yaml.safe_load(f).items()}

    max_offset_index = max(
        list(range(len(task_offsets))),
        key=lambda i: task_offsets[tasks[i]]["offset"]
    )
    total_N = (
        task_offsets[tasks[max_offset_index]]["offset"] 
        + task_offsets[tasks[max_offset_index]]["num_outputs"]
    )
    src_path = os.path.join(args.ckpt_dir, args.ckpt_path)

    pretrained_cfg_src_path = (
        "/weka/dfive-default/helios/checkpoints/favyen"
        "/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000/config.json"
    )
    with open(pretrained_cfg_src_path, "r") as f:
        pretrained_cfg = json.load(f)

    for task, info in task_offsets.items():
        num_outputs = info["num_outputs"]
        offset = info["offset"]
        print(f"\n======== Processing task '{task}' =========")
        print(f"offset={offset}, outputs={num_outputs}, total={total_N}")

        # 1) reload checkpoint per task to keep things isolated
        print(f"Loading checkpoint from {src_path}")
        ckpt = torch.load(src_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        # 2) trim only this task
        trim_state_dict(
            sd, task, offset, num_outputs, N=total_N,
            task_type=task_types[task],
            task_offsets=task_offsets,
        )

        # 3) save per-task checkpoint and config.yaml
        out_dir = args.ckpt_out_dir.format(task=task)
        os.makedirs(out_dir, exist_ok=True)
        dst_path = os.path.join(out_dir, args.ckpt_path)

        if "state_dict" in ckpt:
            ckpt["state_dict"] = sd
            to_save = ckpt
        else:
            to_save = sd

        print(f"\nSaving trimmed checkpoint to {dst_path}")
        torch.save(to_save, dst_path)

        dst_cfg_path = os.path.join(out_dir, "config.yaml")
        with open(dst_cfg_path, "w") as f:
            yaml.dump(cfg, f)
     
        pretrained_dst_cfg_path = os.path.join(out_dir, "../config.json")
        with open(pretrained_dst_cfg_path, "w") as f:
            json.dump(pretrained_cfg, f)

    print("\nAll tasks have been processed and saved.")

if __name__ == "__main__":
    main()
