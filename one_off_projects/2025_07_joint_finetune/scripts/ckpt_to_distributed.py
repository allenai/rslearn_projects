"""Convert rslearn-trained checkpoint to distributed format used by helios.

Use this if trying to do evals on a helios backbone that was trained with rslearn.

Handles the checkpoint conversion, as well as the overrides for the helios model config
(specified if you want to do things like backbone lora, moe, etc in rslearn).
"""

import os
import argparse
import json
import yaml
import importlib
import shutil

from rslearn.train.lightning_module import RestoreConfig
from olmo_core.train.checkpoint import Checkpointer


class DummySaver:

    def __init__(self, state_dict):
        self.state_dict = state_dict

    def state_dict_to_save(self):
        return self.state_dict


def ckpt_to_distributed(*args, **kwargs):

    class HeliosExtractor(kwargs.pop("base_class")):

        def __init__(self, *args, **kwargs):
            save_path = kwargs.pop("save_path")
            ckpt_path = kwargs.pop("ckpt_path")
            work_dir = kwargs.pop("work_dir")

            print("=====================")
            print(f"save_path: {save_path}")
            print(f"ckpt_path: {ckpt_path}")
            print(f"work_dir: {work_dir}")
            print("=====================\n")

            super().__init__(*args, **kwargs)

            ckpt_path = os.path.join(ckpt_path, "checkpoints", "last.ckpt")
            restore_config = RestoreConfig(
                ckpt_path, 
                selector=["state_dict"],
                ignore_prefixes=["model.decoders"],
                remap_prefixes=[("model.encoder.0.model.", "model.encoder.")]
            )
            state_dict = restore_config.get_state_dict()

            dummy_saver = DummySaver(state_dict)
            checkpointer = Checkpointer(work_dir=work_dir)

            shutil.rmtree(save_path)
            checkpointer.save(save_path, dummy_saver, {})
    
    return HeliosExtractor(*args, **kwargs)


def resolve_class_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def deep_merge(base, override):
    for k, v in override.items():
        v_copy = v.copy() if hasattr(v, "copy") else v
        if k.endswith("+"):
            k = k[:-1]
            if k not in base:
                base[k] = []
            base[k].extend(v_copy)
        else:
            if isinstance(v, dict):
                base[k] = deep_merge(base.get(k, {}), v_copy)
            else:
                base[k] = v_copy
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Run directory in args.project (ex: classify_lora_v4)")
    parser.add_argument("--project", type=str, default="2025_08_12_task_embeds", help="Project name/wandb project")
    parser.add_argument("--work_dir", type=str, default="/weka/dfive-default/ryanp/scratch/_tmp")
    parser.add_argument("--save_dir", type=str, default="/weka/dfive-default/ryanp/scratch/distributed_ckpts")
    parser.add_argument("--base_dir", type=str, default="/weka/dfive-default/rslearn-eai/projects")
    args = parser.parse_args()

    finetune_config_path = os.path.join(args.base_dir, args.project, args.run, "checkpoints", "config.yaml")
    with open(finetune_config_path, "r") as f:
        finetune_config = yaml.safe_load(f)
        enc = finetune_config["model"]["init_args"]["model"]["init_args"]["encoder"][0]
        model_config = enc["init_args"]
        base_class = resolve_class_path(enc["class_path"])
        
        print("=====================")
        print(f"base_class: {base_class}")
        print(json.dumps(model_config, indent=2) + "\n")
        print("=====================\n")

    # save the actual distributed checkpoint
    extractor = ckpt_to_distributed(
        base_class=base_class,
        **model_config,
        save_path=os.path.join(args.save_dir, args.run),
        ckpt_path=os.path.join(args.base_dir, args.project, args.run),
        work_dir=args.work_dir,
    )

    # copy over the config.json (helios config, not the rslearn config.yaml)
    base_model_path = os.path.join(model_config["checkpoint_path"], "config.json")
    with open(base_model_path, "r") as f:
        base_model_config = json.load(f)

    overrides = dict(model=model_config["model_overrides"])
    base_model_config = deep_merge(base_model_config, overrides)
    with open(os.path.join(args.save_dir, args.run, "config.json"), "w") as f:
        json.dump(base_model_config, f)
    
    # copy over train/rank0.pt
    train_info_path = os.path.join(model_config["checkpoint_path"], "train", "rank0.pt")    
    train_info_dump = os.path.join(args.save_dir, args.run, "train", "rank0.pt")
    shutil.copy(train_info_path, train_info_dump)

    print(f"done! saved to {os.path.join(args.save_dir, args.run)}")
