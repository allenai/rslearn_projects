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


COMMON: dict[str, Any] = {
    "inputs": EmbeddingInputs.S2,
    "years": [2024],
    "store_path": "gs://bucket/embeddings.zarr",
    "completed_path_template": "gs://bucket/completed_{year}/",
    "queue_name": "user/queue",
    "checkpoint_path": "/fake/ckpt",
    "image_name": "user/image",
    "cluster": ["ai2/jupiter"],
    "model_url": "https://example.invalid/model",
    "source_data": ["https://example.invalid/s2"],
    "artifact_path": "gs://bucket/artifact",
    "pca_store_path": "gs://bucket/pca.zarr",
    "pca_completed_path": "gs://bucket/pca_completed/",
}


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
    monkeypatch.setattr(
        run_all_mod, "init_web_store", lambda **kw: calls.append("init_web")
    )
    monkeypatch.setattr(run_all_mod, "get_web_jobs", lambda **kw: [])

    run_all_mod.run_all(**COMMON, web_min_zoom=12, web_max_zoom=14)
    assert calls == [
        "init",
        "supervise:predict",
        "fit_pca",
        "init_pca",
        "supervise:render_utm_pca",
        "annotate",
        "init_web",
        "supervise:render_web_pca",
        "supervise:render_web_pca",
        "supervise:render_web_pca",
    ]


def test_web_zooms_run_deepest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """A coarse shard reads the four below it, so a shallower zoom must not run first."""
    zooms: list[int] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_pca_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod,
        "supervise",
        lambda **kw: zooms.append(kw["web_zoom"]) if kw.get("web_zoom") else None,
    )
    _stub_paths(monkeypatch, exists=False)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "annotate_pca_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_web_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_web_jobs", lambda **kw: [])

    run_all_mod.run_all(**COMMON, web_min_zoom=10, web_max_zoom=14)
    assert zooms == [14, 13, 12, 11, 10]


def test_web_stage_passes_every_required_supervise_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every argument supervise requires must actually be handed over.

    The stubs elsewhere accept **kw, so an omission is invisible to them and only
    surfaces when the real supervise runs -- which on Beaker means a failed driver
    minutes after launch. This checks the call against the real signature.
    """
    import inspect

    from rslp.large_scale_embeddings import supervise as real_supervise

    required = {
        name
        for name, p in inspect.signature(real_supervise).parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    seen: list[set[str]] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_pca_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "supervise", lambda **kw: seen.append(set(kw)))
    _stub_paths(monkeypatch, exists=False)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "annotate_pca_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_web_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_web_jobs", lambda **kw: [])

    run_all_mod.run_all(**COMMON, web_min_zoom=14, web_max_zoom=14)
    for passed in seen:
        missing = required - passed
        assert not missing, f"supervise call omits {sorted(missing)}"


def test_web_stage_survives_leaked_supervise_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """jsonargparse expands supervise's signature into run_all's CLI.

    Because run_all takes **supervise_kwargs, a supervise-only option such as
    --web_zoom becomes a run_all option and arrives here carrying its default. Passing
    that through while also naming it explicitly raised "got multiple values for
    keyword argument" and failed the driver on the real run.
    """
    seen: list[int] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_pca_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod,
        "supervise",
        lambda **kw: seen.append(kw["web_zoom"]) if kw.get("web_zoom") else None,
    )
    _stub_paths(monkeypatch, exists=False)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "annotate_pca_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_web_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_web_jobs", lambda **kw: [])

    run_all_mod.run_all(
        **COMMON,
        web_min_zoom=13,
        web_max_zoom=14,
        # Exactly what the CLI hands over: supervise's own defaults, unasked for.
        web_zoom=None,
        web_base_zoom=14,
        pending_per_worker=3,
    )
    assert seen == [14, 13]


def test_skip_web_pca_stops_after_annotate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The display pyramid is opt-out, so a run can still produce only the UTM stores."""
    calls: list[str] = []
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_pca_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod, "supervise", lambda **kw: calls.append(f"supervise:{kw['stage']}")
    )
    _stub_paths(monkeypatch, exists=False)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "annotate_pca_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod, "init_web_store", lambda **kw: calls.append("init_web")
    )

    run_all_mod.run_all(**COMMON, skip_web_pca=True)
    assert "init_web" not in calls
    assert "supervise:render_web_pca" not in calls


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
    monkeypatch.setattr(run_all_mod, "init_web_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_web_jobs", lambda **kw: [])

    run_all_mod.run_all(**COMMON, gpus=1)
    assert ("predict", 1) in seen
    assert ("render_utm_pca", 0) in seen
    assert ("render_web_pca", 0) in seen


def test_gdal_env_vars_are_passed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workers get GS_USER_PROJECT without being asked.

    Omitting it fails the run minutes in with an HTTP 400 that names no variable, so it
    is a default rather than something a caller has to remember.
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
    # AWS credentials are secrets, mounted by supervise, not plain env vars. An unsigned
    # request could not serve requester_pays=True, so that flag must not appear here.
    assert "AWS_NO_SIGN_REQUEST" not in seen[0]


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


def test_web_tuning_does_not_reach_the_other_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The short cycle and deep queue are scoped to render_web_pca.

    Both are wrong for the long-job stages: a predict job runs tens of minutes, so a
    120s cycle would re-enumerate markers ~19 times per job and a 64-deep queue would
    leave claims outstanding far longer than a worker's life. Passing them on the
    command line applies them run-wide, which is why they belong in the web kwargs.
    """
    seen: dict[str, dict] = {}
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_pca_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod, "supervise", lambda **kw: seen.setdefault(kw["stage"], kw)
    )
    _stub_paths(monkeypatch, exists=False)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "annotate_pca_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_web_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_web_jobs", lambda **kw: [])

    run_all_mod.run_all(**COMMON, web_min_zoom=14, web_max_zoom=14)

    web = seen[run_all_mod.STAGE_RENDER_WEB_PCA]
    assert web["cycle_seconds"] == run_all_mod.WEB_CYCLE_SECONDS
    assert web["pending_per_worker"] == run_all_mod.WEB_PENDING_PER_WORKER
    for stage in ("predict", run_all_mod.STAGE_RENDER_UTM_PCA):
        # Absent entirely, so supervise applies its own long-job defaults.
        assert "cycle_seconds" not in seen[stage]
        assert "pending_per_worker" not in seen[stage]


def test_explicit_cycle_seconds_still_wins_for_the_web_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """setdefault, not assignment: an operator can still override at launch."""
    seen: dict[str, dict] = {}
    monkeypatch.setattr(run_all_mod, "init_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_pca_store", lambda **kw: None)
    monkeypatch.setattr(
        run_all_mod, "supervise", lambda **kw: seen.setdefault(kw["stage"], kw)
    )
    _stub_paths(monkeypatch, exists=False)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "annotate_pca_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_web_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_web_jobs", lambda **kw: [])

    run_all_mod.run_all(**COMMON, web_min_zoom=14, web_max_zoom=14, cycle_seconds=45)

    assert seen[run_all_mod.STAGE_RENDER_WEB_PCA]["cycle_seconds"] == 45
    assert seen["predict"]["cycle_seconds"] == 45
