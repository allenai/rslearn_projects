"""The window time range the prediction pipeline gives its windows.

Callers pass a reference timestamp (T, T) and let each layer derive its own request
range from time_offset/duration. That worked while the model read timestamps off the
materialized items. OlmoEarthPeriodTimestamps reads the window instead, building its
period grid backwards from the end, so a zero span means zero periods and every tile
fails. These check the window is widened to cover what the layers will request.

Real `DataSourceConfig` objects rather than fakes, so the span follows rslearn's own
window-to-request conversion instead of a second copy of that arithmetic here.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from rslearn.config.dataset import DataSourceConfig

from rslp.large_scale_embeddings.predict_pipeline import _window_span

T = datetime(2025, 1, 1, tzinfo=UTC)


def _dataset(**layers: DataSourceConfig | None) -> SimpleNamespace:
    """A stand-in dataset carrying only the layer data source configs.

    Args:
        layers: layer name to its data source config, or None for a layer with none.

    Returns:
        an object with the `.layers` the helper reads.
    """
    return SimpleNamespace(
        layers={
            name: SimpleNamespace(data_source=source) for name, source in layers.items()
        }
    )


def _source(
    days: int | None = None, offset_days: int | None = None
) -> DataSourceConfig:
    """A data source config with an optional duration and offset."""
    return DataSourceConfig(
        class_path="x",
        duration=timedelta(days=days) if days else None,
        time_offset=timedelta(days=offset_days) if offset_days else None,
    )


def test_window_covers_the_latest_end_any_layer_requests() -> None:
    """The window must reach the furthest point any layer will ask for."""
    got = _window_span(_dataset(a=_source(365), b=_source(180)), (T, T))
    assert got == (T, T + timedelta(days=365)), got


def test_a_layers_time_offset_extends_the_window() -> None:
    """An offset layer requests later imagery, so the window has to reach it.

    Reading `duration` alone would stop at T+365d and leave the final month of an
    offset layer outside the window, which is the kind of thing that shows up as a
    quietly truncated period grid rather than an error.
    """
    got = _window_span(_dataset(a=_source(365, offset_days=30)), (T, T))
    assert got == (T, T + timedelta(days=395)), got


def test_a_layer_without_a_data_source_is_ignored() -> None:
    """The output layer has no data source, so it must not break the lookup."""
    got = _window_span(_dataset(output=None, s2=_source(365)), (T, T))
    assert got == (T, T + timedelta(days=365)), got


def test_no_durations_leaves_the_range_untouched() -> None:
    """A config that declares no durations must behave exactly as it did before."""
    assert _window_span(_dataset(a=_source(), b=None), (T, T)) == (T, T)


def test_the_real_config_fits_the_models_period_count() -> None:
    """The shipped config must give the model the periods it is configured for.

    Checked against the real files rather than a fixture: the duration lives in the
    dataset config and the period count in the model config, and the failure mode when
    they disagree is every tile raising, not a quiet degradation.
    """
    root = Path("data/large_scale_embeddings")
    cfg = json.loads((root / "s2_s1_landsat_distilled.json").read_text())

    def _days(raw: str | None) -> int | None:
        if not raw:
            return None
        assert raw.endswith("d"), f"unhandled duration format {raw!r}"
        return int(raw[:-1])

    layers: dict[str, DataSourceConfig | None] = {}
    for name, layer in cfg["layers"].items():
        src = layer.get("data_source")
        layers[name] = (
            _source(_days(src.get("duration")), _days(src.get("time_offset")))
            if src
            else None
        )

    start, end = _window_span(_dataset(**layers), (T, T))
    span = end - start
    assert span > timedelta(0), "the window must not be a zero-length range"

    model_yaml = (root / "s2_s1_landsat_distilled.yaml").read_text()
    if "OlmoEarthPeriodTimestamps" in model_yaml:
        import yaml

        init = yaml.safe_load(model_yaml)["model"]["init_args"]["model"]["init_args"][
            "encoder"
        ][0]["init_args"]
        period = init["period_duration"]
        days = int(period.split(" days")[0]) if "days" in period else None
        assert days, f"unhandled period_duration format {period!r}"
        fits = span // timedelta(days=days)
        assert fits >= init["max_matches"], (
            f"window spans {span} which fits {fits} periods of {days}d, "
            f"short of max_matches={init['max_matches']}"
        )


def test_the_pipeline_gives_windows_the_widened_range() -> None:
    """The Window must be built from the widened range, not the raw argument.

    Checked at the call site because the helper being correct is worth nothing if the
    pipeline keeps passing the reference timestamp. Reverting that one line is a silent
    change: materialization is identical, and only the model notices, by raising on
    every tile.
    """
    import ast
    import importlib
    import inspect

    # importlib, not `from ... import predict_pipeline`: the package re-exports the
    # function of that name, so a plain import hands getsource one function body
    # instead of the module, and the Window call is in a different function.
    mod = importlib.import_module("rslp.large_scale_embeddings.predict_pipeline")
    assert inspect.ismodule(mod), "expected the module, got the re-exported function"

    tree = ast.parse(inspect.getsource(mod))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Window"
    ]
    assert calls, "no Window(...) construction found"

    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "time_range" in kw, "Window built without a time_range"
        value = kw["time_range"]
        assert isinstance(value, ast.Name) and value.id == "window_time_range", (
            "Window is built from the raw reference timestamp; "
            "OlmoEarthPeriodTimestamps needs the widened span"
        )
