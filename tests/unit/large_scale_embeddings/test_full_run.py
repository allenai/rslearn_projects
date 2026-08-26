"""Unit tests for rslp.large_scale_embeddings.run_all.

The behaviour under test is the refusal to advance on partial input. supervise returns
normally when it exhausts its cycles, so a pipeline that trusted that return would fit a
PCA basis on a partly-written archive and render colours that look plausible and are
wrong. Nothing downstream detects that, so it has to fail here.
"""

from typing import Any

import pytest

import rslp.large_scale_embeddings.full_run as run_all_mod
from rslp.large_scale_embeddings.predict_pipeline import EmbeddingInputs


class _StubPath:
    """Stands in for UPath so existence checks never touch a real filesystem."""

    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists


def _stub_paths(monkeypatch: pytest.MonkeyPatch, *, exists: bool) -> None:
    monkeypatch.setattr(run_all_mod, "UPath", lambda _path: _StubPath(exists))


COMMON: dict[str, Any] = dict(
    inputs=EmbeddingInputs.S2,
    years=[2024],
    store_path="gs://bucket/embeddings.zarr",
    completed_path_template="gs://bucket/completed_{year}/",
    queue_name="user/queue",
    checkpoint_path="/fake/ckpt",
    image_name="user/image",
    cluster=["ai2/jupiter"],
    model_url="https://example.invalid/model",
    source_data=["https://example.invalid/s2"],
    artifact_path="gs://bucket/artifact",
    pca_store_path="gs://bucket/pca.zarr",
    pca_completed_path="gs://bucket/pca_completed/",
)


def test_predict_shortfall_stops_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks left unpredicted must abort before fit_pca runs."""
    fitted: list[str] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "supervise", lambda **kw: None)
    _stub_paths(monkeypatch, exists=True)
    # Two blocks still outstanding after supervise returned.
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [["a"], ["b"]])
    monkeypatch.setattr(
        run_all_mod, "fit_pca", lambda **kw: fitted.append(kw["artifact_path"])
    )

    with pytest.raises(RuntimeError, match="refusing to fit a PCA basis"):
        run_all_mod.run_all(**COMMON)
    assert fitted == [], "fit_pca must not run on a partly-written archive"


def test_render_shortfall_stops_before_annotating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blocks left unrendered must abort before the store is annotated.

    Annotating records basis provenance onto every level, which is what tells a reader
    the pyramid is finished and usable.
    """
    annotated: list[str] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_pca_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "supervise", lambda **kw: None)
    _stub_paths(monkeypatch, exists=True)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [["x"]])
    monkeypatch.setattr(
        run_all_mod,
        "annotate_pca_store",
        lambda **kw: annotated.append(kw["pca_store_path"]),
    )

    with pytest.raises(RuntimeError, match="unrendered"):
        run_all_mod.run_all(**COMMON)
    assert annotated == [], "a partly-rendered store must not be annotated"


def test_all_stages_run_in_order_when_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path runs predict, fit, render and annotate, in that order."""
    calls: list[str] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: calls.append("init"))
    monkeypatch.setattr(
        run_all_mod, "init_pca_store", lambda **kw: calls.append("init_pca")
    )
    monkeypatch.setattr(
        run_all_mod, "supervise", lambda **kw: calls.append(f"supervise:{kw['stage']}")
    )
    _stub_paths(monkeypatch, exists=False)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: calls.append("fit_pca"))
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [])
    monkeypatch.setattr(
        run_all_mod, "annotate_pca_store", lambda **kw: calls.append("annotate")
    )

    run_all_mod.run_all(**COMMON)
    assert calls == [
        "init",
        "supervise:predict",
        "fit_pca",
        "init_pca",
        "supervise:render_pca",
        "annotate",
    ]


def test_skip_pca_stops_after_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    """skip_pca produces embeddings only, without fitting or rendering."""
    calls: list[str] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod, "supervise", lambda **kw: calls.append(f"supervise:{kw['stage']}")
    )
    _stub_paths(monkeypatch, exists=True)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: calls.append("fit_pca"))

    run_all_mod.run_all(**COMMON, skip_pca=True)
    assert calls == ["supervise:predict"]


def test_render_stage_defaults_to_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The render stage asks for no GPU: it runs no model, so a GPU would sit idle."""
    seen: list[tuple[str, object]] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_pca_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod,
        "supervise",
        lambda **kw: seen.append((kw["stage"], kw.get("gpus"))),
    )
    _stub_paths(monkeypatch, exists=True)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "annotate_pca_store", lambda **kw: None)

    run_all_mod.run_all(**COMMON, gpus=1)
    assert ("predict", 1) in seen
    assert ("render_pca", 0) in seen


def test_gdal_env_vars_are_passed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workers get the GDAL settings the data sources need without being asked.

    Omitting either fails the run minutes in with an error naming neither variable, so
    they are defaults rather than something a caller has to remember.
    """
    seen: list[dict] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod, "supervise", lambda **kw: seen.append(kw["worker_env_vars"])
    )
    _stub_paths(monkeypatch, exists=True)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])

    run_all_mod.run_all(**COMMON, skip_pca=True)
    assert seen[0]["GS_USER_PROJECT"] == "earthsystem-dev-c3po"
    assert seen[0]["AWS_NO_SIGN_REQUEST"] == "YES"


def test_caller_can_override_a_gdal_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit value wins over the default, and the other default survives."""
    seen: list[dict] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod, "supervise", lambda **kw: seen.append(kw["worker_env_vars"])
    )
    _stub_paths(monkeypatch, exists=True)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])

    run_all_mod.run_all(
        **COMMON, skip_pca=True, worker_env_vars={"GS_USER_PROJECT": "other-project"}
    )
    assert seen[0]["GS_USER_PROJECT"] == "other-project"
    assert seen[0]["AWS_NO_SIGN_REQUEST"] == "YES"
