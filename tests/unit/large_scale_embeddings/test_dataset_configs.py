"""Check the dataset configs' init_args against the data sources they name.

An unaccepted init_args key only fails at dataset-prepare time inside a worker, so the
signatures are read statically here.
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "data" / "large_scale_embeddings"
CONFIGS = sorted(CONFIG_DIR.glob("s2*.json"))


def _olmoearth_run_root() -> Path | None:
    """The directory holding the olmoearth_run package, installed or vendored."""
    spec = importlib.util.find_spec("olmoearth_run")
    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).parent
    vendored = REPO / "docker_build" / "olmoearth_run" / "src"
    return vendored if (vendored / "olmoearth_run").exists() else None


SOURCE_ROOT = _olmoearth_run_root()

pytestmark = pytest.mark.skipif(
    SOURCE_ROOT is None, reason="olmoearth_run is neither installed nor vendored"
)


def _accepted_params(class_path: str) -> tuple[set[str], bool] | None:
    """Read the __init__ parameters of a data source named by dotted class path."""
    assert SOURCE_ROOT is not None
    module_path, cls_name = class_path.rsplit(".", 1)
    path = (SOURCE_ROOT / Path(*module_path.split("."))).with_suffix(".py")
    if not path.exists():
        return None
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
        assert (
            accepted is not None
        ), f"cannot locate {data_source['class_path']} under {SOURCE_ROOT}"
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
