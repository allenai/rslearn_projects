"""Generate the 16 fable_20260710 experiment configs from the pass20_v1_2 baseline.

Run from the rslearn_projects root:
    python data/change_finder_v2/lcc_model/fable_20260710/gen_configs.py

Each config is a full standalone YAML (project 2026_07_10_lcc_fable) differing from
data/change_finder_v2/lcc_model/config_pass20_v1_2.yaml only in:
- project_name / run_name
- the model class and its decoder options
- optionally the train-time sampler (quarterly dropout augmentation)
- an added val metric (neg_max_score) tracking hallucination on negative windows
"""

import copy
import os

import yaml

BASE_CONFIG = "data/change_finder_v2/lcc_model/config_pass20_v1_2.yaml"
OUT_DIR = "data/change_finder_v2/lcc_model/fable_20260710"
PROJECT_NAME = "2026_07_10_lcc_fable"

FABLE_MODEL = "rslp.change_finder_v2.lcc_model.model_fable_20260710.FableChangeModel"
FABLE_SAMPLER = "rslp.change_finder_v2.lcc_model.model_fable_20260710.FableSampler"
NEG_METRIC = (
    "rslp.change_finder_v2.lcc_model.model_fable_20260710.NegativeWindowMaxScore"
)

# name -> dict with optional keys:
#   model: extra init_args for FableChangeModel (presence selects FableChangeModel;
#          absence keeps the baseline SinglePassChangeModel)
#   baseline: extra init_args for the baseline SinglePassChangeModel
#   qdrop: quarterly_dropout probability for the train sampler
VARIANTS = {
    # Controls (baseline architecture, re-trained in this project for comparison).
    "base_mean": {},
    "base_diff": {"baseline": {"temporal_aggregation": "diff"}},
    "base_mean_qdrop": {"qdrop": 0.25},
    # Temporal centering: binary head sees only temporal residual statistics.
    "centered": {"model": {"binary_mode": "centered"}},
    "centered_season": {"model": {"binary_mode": "centered", "season_embed": True}},
    "centered_qdrop": {"model": {"binary_mode": "centered"}, "qdrop": 0.25},
    # Breakpoint scan: change = before/after contrast at some split.
    "bp_max": {"model": {"binary_mode": "breakpoint"}},
    "bp_lse": {"model": {"binary_mode": "breakpoint", "evidence_pool": "lse"}},
    "bp_abs": {"model": {"binary_mode": "breakpoint", "contrast": "abs"}},
    "bp_season": {"model": {"binary_mode": "breakpoint", "season_embed": True}},
    "bp_qdrop": {"model": {"binary_mode": "breakpoint"}, "qdrop": 0.25},
    "bp_season_qdrop": {
        "model": {"binary_mode": "breakpoint", "season_embed": True},
        "qdrop": 0.25,
    },
    "bp_big": {
        "model": {
            "binary_mode": "breakpoint",
            "scorer_hidden": 512,
            "evidence_stages": [[[512, 3]], [[256, 3]], [[128, 3]]],
        }
    },
    # Hybrid: breakpoint + centered evidence concatenated.
    "hybrid": {"model": {"binary_mode": "hybrid"}},
    "hybrid_season": {"model": {"binary_mode": "hybrid", "season_embed": True}},
    "hybrid_season_qdrop": {
        "model": {"binary_mode": "hybrid", "season_embed": True},
        "qdrop": 0.25,
    },
    # Round 2 (2026-07-11): bp_abs led the first sweep on val AUROC/PRAUC, so fill
    # in its season/qdrop cells (missing from the original 16).
    "bp_abs_season": {
        "model": {"binary_mode": "breakpoint", "contrast": "abs", "season_embed": True}
    },
    "bp_abs_qdrop": {
        "model": {"binary_mode": "breakpoint", "contrast": "abs"},
        "qdrop": 0.25,
    },
    "bp_abs_season_qdrop": {
        "model": {"binary_mode": "breakpoint", "contrast": "abs", "season_embed": True},
        "qdrop": 0.25,
    },
}


def main() -> None:
    """Generate one YAML per variant."""
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)

    for name, spec in VARIANTS.items():
        cfg = copy.deepcopy(base)
        cfg["project_name"] = PROJECT_NAME
        cfg["run_name"] = name

        model_cfg = cfg["model"]["init_args"]["model"]
        if "model" in spec:
            model_cfg["class_path"] = FABLE_MODEL
            model_cfg["init_args"].update(spec["model"])
        elif "baseline" in spec:
            model_cfg["init_args"].update(spec["baseline"])

        if "qdrop" in spec:
            train_transforms = cfg["data"]["init_args"]["train_config"]["transforms"]
            assert "SinglePassSampler" in train_transforms[0]["class_path"]
            train_transforms[0] = {
                "class_path": FABLE_SAMPLER,
                "init_args": {
                    "deterministic": False,
                    "quarterly_dropout": spec["qdrop"],
                },
            }

        binary_task = cfg["data"]["init_args"]["task"]["init_args"]["tasks"]["binary"]
        binary_task["init_args"]["other_metrics"]["neg_max_score"] = {
            "class_path": NEG_METRIC
        }

        out_path = os.path.join(OUT_DIR, f"{name}.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
