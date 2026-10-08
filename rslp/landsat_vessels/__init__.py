"""Landsat vessel detection project."""

from typing import Any

# Import predict_pipeline lazily so the feedback tools don't load torch.

__all__ = ["predict_pipeline", "workflows"]


def __getattr__(name: str) -> Any:  # noqa: D401 - module-level lazy attribute hook
    if name in ("workflows", "predict_pipeline"):
        from .predict_pipeline import predict_pipeline

        globals()["predict_pipeline"] = predict_pipeline
        globals()["workflows"] = {"predict": predict_pipeline}
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
