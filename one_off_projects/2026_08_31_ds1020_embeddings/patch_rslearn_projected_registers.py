"""Teach rslearn's OlmoEarth wrapper to return the detached student's embedding.

The distilled candidate (regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1) is a 768-wide
encoder with a per-cell Linear(768, 128) student on the register bottleneck. The
shipped 128-dim embedding is that student's output, which the model exposes as
``projected_registers``; ``use_register_bottleneck_output`` returns the 768-dim
``registers`` instead. Taking the first 128 of those would not be the distilled
embedding, so this is a correctness fix, not a convenience.

rslearn has no flag for it yet (checked against master 68a9009f), so this patches
the installed module to honour an environment variable:

    OE_PROJECTED_REGISTER_DIM=128   # use projected_registers[..., :128]

Unset, behaviour is unchanged. Matches olmoearth_pretrain's eval path, which reads
``projected_registers`` and slices a Matryoshka prefix
(olmoearth_pretrain/evals/eval_wrapper.py::_pool_registers).

The proper home for this is an rslearn flag; this exists so a one-off run does not
wait on that.
"""

import pathlib

import rslearn.models.olmoearth_pretrain.model as model_module

MARKER = "OE_PROJECTED_REGISTER_DIM"

# Anchor on the single line that reads the registers, not the whole block: the
# installed release can differ from master in comments and error text.
ANCHOR = '            registers = model_output["registers"]  # [B, n_h*n_w, D]'

INSERT = """
            # Patched: with OE_PROJECTED_REGISTER_DIM set, return the detached
            # student's projection instead of the raw registers, sliced to that
            # width (a Matryoshka prefix), which is what the distilled arm ships.
            _proj_dim = os.environ.get("OE_PROJECTED_REGISTER_DIM")
            if _proj_dim:
                if "projected_registers" not in model_output:
                    raise ValueError(
                        "OE_PROJECTED_REGISTER_DIM is set but the model output has "
                        "no 'projected_registers'; this checkpoint has no detached "
                        "register student"
                    )
                registers = model_output["projected_registers"][..., : int(_proj_dim)]"""


def main() -> None:
    path = pathlib.Path(model_module.__file__)
    source = path.read_text()
    if MARKER in source:
        print(f"{path}: already patched")
        return
    if ANCHOR not in source:
        raise SystemExit(
            f"{path}: register-bottleneck read not found; rslearn changed shape, "
            "re-check this patch instead of running with silently wrong embeddings"
        )
    source = source.replace(ANCHOR, ANCHOR + INSERT, 1)
    if "\nimport os\n" not in source:
        source = source.replace("\nimport warnings\n", "\nimport os\nimport warnings\n", 1)
    path.write_text(source)
    print(f"{path}: patched to honour {MARKER}")


if __name__ == "__main__":
    main()
