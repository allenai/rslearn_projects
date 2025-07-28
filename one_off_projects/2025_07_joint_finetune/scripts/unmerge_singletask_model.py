#!/usr/bin/env python
"""Prepare a model trained across various instances of a task for a single dataset."""

import os
import re
import argparse
import yaml
import torch
from typing import Any, Dict, List


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


def slice_tensor(tensor: torch.Tensor, keep_idx: List[int]) -> torch.Tensor:
    """Index the first dim of tensor by keep_idx and clone

    Args:
        tensor (torch.Tensor): The tensor to index.
        keep_idx (List[int]): The indices to keep.
    Returns:
        torch.Tensor: The indexed tensor.
    """
    return tensor[keep_idx].clone()


def trim_state_dict(sd: Dict[str, torch.Tensor], task: str, i: int, n: int, N: int, task_type: str) -> None:
    """
    Mutates sd in-place, slicing out i:i+n from the relevant head of the given task.

    Args:
        sd (Dict[str, torch.Tensor]): The state dict to trim.
        task (str): The task name.
        i (int): Start index for trimming.
        n (int): Number of outputs to keep.
        N (int): Original total number of outputs.
        task_type (str): The task type (detect/classify/segment).
    """
    print(f"\n-> Trimming task '{task}': keep indices [{i}:{i+n}] out of {N}")

    if task_type == "detect":
        cls_w = next(k for k in sd if k.endswith("box_predictor.cls_score.weight"))
        cls_b = cls_w.replace(".weight", ".bias")
        reg_w = next(k for k in sd if k.endswith("box_predictor.bbox_pred.weight"))
        reg_b = reg_w.replace(".weight", ".bias")

        assert sd[cls_w].shape[0] == N, f"Expected {N} cls_score rows, got {sd[cls_w].shape[0]}"
        assert sd[reg_w].shape[0] == 4 * N, f"Expected {4*N} bbox rows, got {sd[reg_w].shape[0]}"

        keep = [0] + list(range(i, i + n))  # preserve background index 0
        keep_reg: List[int] = []
        for k in keep:
            keep_reg += list(range(4*k, 4*k + 4))

        print(f"  - cls_score: original {sd[cls_w].shape}, new rows={len(keep)}")
        sd[cls_w] = slice_tensor(sd[cls_w], keep)
        sd[cls_b] = slice_tensor(sd[cls_b], keep)

        print(f"  - bbox_pred: original {sd[reg_w].shape}, new rows={len(keep_reg)}")
        sd[reg_w] = slice_tensor(sd[reg_w], keep_reg)
        sd[reg_b] = slice_tensor(sd[reg_b], keep_reg)

    elif task_type == "classify":
        wkey = next(k for k in sd if k.endswith("ClassificationHead.0.output_layer.weight"))
        bkey = wkey.replace(".weight", ".bias")
        assert sd[wkey].shape[0] == N, f"Expected {N} rows, got {sd[wkey].shape[0]}"

        keep = list(range(i, i + n))
        print(f"  - output_layer: original {sd[wkey].shape}, new rows={len(keep)}")
        sd[wkey] = slice_tensor(sd[wkey], keep)
        sd[bkey] = slice_tensor(sd[bkey], keep)

    elif task_type == "segment":
        pattern = re.compile(r".*SegmentationHead.*layers\.\d+\.\d+\.weight$")
        w_keys = [k for k in sd if pattern.match(k)]
        if not w_keys:
            raise KeyError(f"No SegmentationHead weight keys found for task '{task}'")
        w_keys = sorted(w_keys, key=lambda x: int(x.split(".")[-2]))
        wkey = w_keys[-1]
        bkey = wkey.replace(".weight", ".bias")
        assert sd[wkey].shape[0] == N, f"Expected {N} rows, got {sd[wkey].shape[0]}"

        keep = list(range(i, i + n))
        print(f"  - {wkey}: original {sd[wkey].shape}, new rows={len(keep)}")
        sd[wkey] = slice_tensor(sd[wkey], keep)
        sd[bkey] = slice_tensor(sd[bkey], keep)

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
        trim_state_dict(sd, task, offset, num_outputs, N=total_N, task_type=task_types[task])

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

    print("\nAll tasks have been processed and saved.")

if __name__ == "__main__":
    main()
