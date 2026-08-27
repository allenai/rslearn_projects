"""Generate the 45 SimpleConv sweep override configs next to base.yaml.

Each emitted YAML is a small override that is merged on top of ``base.yaml`` via
two ``--config`` flags:

    rslearn model fit --config base.yaml --config <variant>.yaml

The override fully specifies the ``model.init_args.model`` block (class_path +
init_args) plus ``run_name`` (and ``batch_size`` for the heavy 3D/256 cells so
they fit in memory). The generated files are intentionally NOT committed; only
this script and ``base.yaml`` are.

Sweep:
- base grid (36): conv{2d,3d} x layers{4,8,12} x dim{64,128,256} x head{attn,rnn}
- self-attn (9): conv=3d, head=attn, 2 temporal self-attn layers, over the same
  9 layer/dim combos.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONV_TYPES = ["2d", "3d"]
NUM_LAYERS = [4, 8, 12]
EMBED_DIMS = [64, 128, 256]
HEAD_TYPES = ["attn", "rnn"]

MODEL_CLASS = "rslp.change_finder_v2.lcc_model.model_simpleconv.SimpleConvChangeModel"


def _is_heavy(conv_type: str, num_layers: int, embedding_dim: int) -> bool:
    """Whether the full-res 3D activations need a reduced batch size."""
    return conv_type == "3d" and embedding_dim == 256 and num_layers >= 8


def _variant(
    conv_type: str,
    num_layers: int,
    embedding_dim: int,
    head_type: str,
    num_selfattn: int,
    run_name: str,
) -> dict:
    """Build one override config dict."""
    cfg: dict = {
        "model": {
            "init_args": {
                "model": {
                    "class_path": MODEL_CLASS,
                    "init_args": {
                        "conv_type": conv_type,
                        "num_conv_layers": num_layers,
                        "embedding_dim": embedding_dim,
                        "head_type": head_type,
                        "num_temporal_selfattn_layers": num_selfattn,
                        "num_classes_binary": 3,
                        "num_classes_src": 13,
                        "num_classes_dst": 13,
                        "num_timesteps": 20,
                        "in_channels": 12,
                        "binary_loss_weight": 2.0,
                    },
                }
            }
        },
        "run_name": run_name,
    }
    if _is_heavy(conv_type, num_layers, embedding_dim):
        cfg["data"] = {"init_args": {"batch_size": 4}}
    return cfg


def main() -> None:
    """Write all 45 override configs into this script's directory."""
    out_dir = Path(__file__).parent
    variants: list[tuple[str, dict]] = []

    # Base grid (36).
    for conv_type in CONV_TYPES:
        for num_layers in NUM_LAYERS:
            for embedding_dim in EMBED_DIMS:
                for head_type in HEAD_TYPES:
                    run_name = (
                        f"simpleconv_{conv_type}_{num_layers}l_"
                        f"{embedding_dim}d_{head_type}"
                    )
                    variants.append(
                        (
                            run_name,
                            _variant(
                                conv_type,
                                num_layers,
                                embedding_dim,
                                head_type,
                                0,
                                run_name,
                            ),
                        )
                    )

    # Self-attention variants (9): 3d + attn + 2 temporal self-attn layers.
    for num_layers in NUM_LAYERS:
        for embedding_dim in EMBED_DIMS:
            run_name = f"simpleconv_3d_{num_layers}l_{embedding_dim}d_attn_sa2"
            variants.append(
                (
                    run_name,
                    _variant("3d", num_layers, embedding_dim, "attn", 2, run_name),
                )
            )

    for run_name, cfg in variants:
        path = out_dir / f"{run_name}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote {len(variants)} override configs to {out_dir}")


if __name__ == "__main__":
    main()
