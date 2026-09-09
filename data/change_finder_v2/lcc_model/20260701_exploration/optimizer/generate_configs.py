"""Generate the optimizer/freeze sweep override configs next to base.yaml.

Emits 33 override YAMLs, each merged on top of base.yaml via two --config flags:

    rslearn model fit --config base.yaml --config <variant>.yaml

Primary family (LayerDecayAdamW + SimpleFreeze), 27 runs:
  - unfreeze{none,5} x layer_decay_rate{1.0,0.9,0.8,0.65} x lr{5e-5,1e-4,3e-4} = 24
  - frozen x lr{5e-5,1e-4,3e-4} = 3 (fixed decay=1.0; decay is a no-op when the
    encoder never trains)

Factor family (uniform AdamW + FreezeUnfreeze), 6 runs:
  - lr{1e-4,3e-4} x unfreeze_lr_factor{1,10,100}, unfreeze_at_epoch=5

The generated files are NOT committed; only this script and base.yaml are.
"""

from __future__ import annotations

from pathlib import Path

import yaml

LDECAY_CLASS = "rslearn.models.olmoearth_pretrain.optimizer.LayerDecayAdamW"
ADAMW_CLASS = "rslearn.train.optimizer.AdamW"
SIMPLEFREEZE_CLASS = "rslearn.models.olmoearth_pretrain.optimizer.SimpleFreeze"
FREEZEUNFREEZE_CLASS = "rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze"

MODULE_SELECTOR = ["model", "encoder"]
FROZEN_EPOCH = 10000  # never unfreezes within max_epochs (linear-probe)

# (label, value) for learning rates and decay rates.
LRS = [("5e-5", 5e-5), ("1e-4", 1e-4), ("3e-4", 3e-4)]
DECAYS = [("1.0", 1.0), ("0.9", 0.9), ("0.8", 0.8), ("0.65", 0.65)]
FACTOR_LRS = [("1e-4", 1e-4), ("3e-4", 3e-4)]
FACTORS = [1, 10, 100]


def _ldecay_optimizer(lr: float, decay: float) -> dict:
    return {
        "model": {
            "init_args": {
                "optimizer": {
                    "class_path": LDECAY_CLASS,
                    "init_args": {
                        "lr": lr,
                        "layer_decay_rate": decay,
                        "num_layers": 12,
                        "encoder_prefix": "model.encoder",
                    },
                }
            }
        }
    }


def _simplefreeze_callback(unfreeze_at_epoch: int) -> dict:
    return {
        "trainer": {
            "callbacks+": [
                {
                    "class_path": SIMPLEFREEZE_CLASS,
                    "init_args": {
                        "module_selector": MODULE_SELECTOR,
                        "unfreeze_at_epoch": unfreeze_at_epoch,
                    },
                }
            ]
        }
    }


def main() -> None:
    """Write all 33 override configs into this script's directory."""
    out_dir = Path(__file__).parent
    configs: dict[str, dict] = {}

    # Primary family: none / 5 across decay x lr.
    for uf_label, uf_epoch in [("none", None), ("5", 5)]:
        for d_label, d_val in DECAYS:
            for lr_label, lr_val in LRS:
                name = f"opt_lr{lr_label}_decay{d_label}_uf{uf_label}"
                cfg = _ldecay_optimizer(lr_val, d_val)
                if uf_epoch is not None:
                    cfg.update(_simplefreeze_callback(uf_epoch))
                cfg["run_name"] = name
                configs[name] = cfg

    # Primary family: frozen (decay fixed at 1.0) across lr.
    for lr_label, lr_val in LRS:
        name = f"opt_lr{lr_label}_frozen"
        cfg = _ldecay_optimizer(lr_val, 1.0)
        cfg.update(_simplefreeze_callback(FROZEN_EPOCH))
        cfg["run_name"] = name
        configs[name] = cfg

    # Factor family: uniform AdamW + FreezeUnfreeze(unfreeze_lr_factor).
    for lr_label, lr_val in FACTOR_LRS:
        for factor in FACTORS:
            name = f"optf_lr{lr_label}_uf5_factor{factor}"
            cfg = {
                "model": {
                    "init_args": {
                        "optimizer": {
                            "class_path": ADAMW_CLASS,
                            "init_args": {"lr": lr_val},
                        }
                    }
                },
                "trainer": {
                    "callbacks+": [
                        {
                            "class_path": FREEZEUNFREEZE_CLASS,
                            "init_args": {
                                "module_selector": MODULE_SELECTOR,
                                "unfreeze_at_epoch": 5,
                                "unfreeze_lr_factor": factor,
                            },
                        }
                    ]
                },
                "run_name": name,
            }
            configs[name] = cfg

    for name, cfg in configs.items():
        with (out_dir / f"{name}.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote {len(configs)} override configs to {out_dir}")


if __name__ == "__main__":
    main()
