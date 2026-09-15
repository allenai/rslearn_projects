"""Landsat vessel detection project."""

from typing import Any

# ``predict_pipeline`` pulls in torch + rslearn models, which are slow to import
# (especially off networked storage). Import it lazily via PEP 562 so lightweight
# submodules -- e.g. the feedback tooling under ``rslp.landsat_vessels.feedback`` -- can
# import the package without paying for the full model stack. ``rslp.main`` reads
# ``workflows`` only when a workflow is actually run, which triggers the import then.

__all__ = ["predict_pipeline", "workflows"]


def __getattr__(name: str) -> Any:  # noqa: D401 - module-level lazy attribute hook
    if name in ("workflows", "predict_pipeline"):
        from .predict_pipeline import predict_pipeline

        globals()["predict_pipeline"] = predict_pipeline
        globals()["workflows"] = {"predict": predict_pipeline}
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
