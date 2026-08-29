"""Validate the dataset configs against the data-source signatures they name.

jsonargparse rejects an unaccepted ``init_args`` key at dataset-prepare time, inside a
worker, minutes into a job. Landsat's source takes no ``cache_dir`` while Sentinel-2's
does, and copying one layer's init_args to the other is an easy mistake that costs an
image rebuild to discover. Signatures are read statically so this needs none of
olmoearth_run's runtime dependencies.
"""

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "data" / "large_scale_embeddings"
SOURCE_DIR = (
    REPO
    / "docker_build"
    / "olmoearth_run"
    / "src"
    / "olmoearth_run"
    / "runner"
    / "tools"
    / "rslearn_data_sources"
)

CONFIGS = sorted(CONFIG_DIR.glob("s2*.json"))

# docker_build/ is gitignored, so these sources exist only in a local dev checkout.
pytestmark = pytest.mark.skipif(
    not SOURCE_DIR.exists(),
    reason="olmoearth_run sources not checked out under docker_build/",
)


def _accepted_params(class_path: str) -> tuple[set[str], bool] | None:
    """Read the __init__ parameters of a data source named by dotted class path."""
    module_path, cls_name = class_path.rsplit(".", 1)
    rel = Path(*module_path.split(".")[3:]).with_suffix(".py")
    path = SOURCE_DIR.parent.parent.parent.parent / "olmoearth_run" / rel
    if not path.exists():
        # Fall back to a direct search, since the package layout may shift.
        matches = list(SOURCE_DIR.rglob(f"{module_path.rsplit('.', 1)[-1]}.py"))
        if not matches:
            return None
        path = matches[0]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    a = fn.args
                    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} - {
                        "self"
                    }
                    return names, bool(a.kwarg)
    return None


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_init_args_are_accepted_by_the_data_source(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    checked = 0
    for layer_name, layer in config["layers"].items():
        data_source = layer.get("data_source")
        if not data_source:
            continue
        accepted = _accepted_params(data_source["class_path"])
        assert accepted is not None, (
            f"cannot locate {data_source['class_path']} under {SOURCE_DIR}"
        )
        names, takes_kwargs = accepted
        if takes_kwargs:
            continue
        given = set(data_source.get("init_args", {}))
        rejected = given - names
        assert not rejected, (
            f"{config_path.name} layer {layer_name!r} passes {sorted(rejected)} to "
            f"{data_source['class_path']}, which accepts {sorted(names)}"
        )
        checked += 1
    assert checked, f"{config_path.name} declared no data sources to check"
