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
from rslp.large_scale_embeddings.supervise import (
    CycleConfig,
    ModelConfig,
    PcaConfig,
    WorkerConfig,
)


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
    "model": ModelConfig(checkpoint_path="/fake/ckpt"),
    "worker": WorkerConfig(image_name="user/image", cluster=["ai2/jupiter"]),
    "model_url": "https://example.invalid/model",
    "source_data": ["https://example.invalid/s2"],
    "pca": PcaConfig(
        artifact_path="gs://bucket/artifact",
        store_path="gs://bucket/pca.zarr",
        completed_path="gs://bucket/pca_completed/",
    ),
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
        lambda **kw: (
            zooms.append(kw["pca"].web_zoom)
            if kw["stage"] == run_all_mod.STAGE_RENDER_WEB_PCA
            else None
        ),
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


def test_run_all_takes_no_open_kwargs() -> None:
    """`run_all` must not accept **kwargs, or jsonargparse leaks options into it again.

    It used to take **supervise_kwargs, which made jsonargparse expand supervise's whole
    signature into run_all's CLI. Every supervise option then arrived here already set
    to its default, indistinguishable from one a person typed, so the web stage's own
    tuning had to be assigned rather than defaulted and a `setdefault` silently did
    nothing. Config objects close that off only while the open kwargs stay gone.
    """
    import inspect

    kinds = [p.kind for p in inspect.signature(run_all_mod.run_all).parameters.values()]
    assert inspect.Parameter.VAR_KEYWORD not in kinds, (
        "run_all accepts **kwargs again, so supervise's options will leak into its CLI"
    )


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
        lambda **kw: seen.append((kw["stage"], kw["worker"].gpus)),
    )
    _stub_paths(monkeypatch, exists=True)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "fit_pca", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_render_jobs", lambda **kw: [])
    monkeypatch.setattr(run_all_mod, "annotate_pca_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "init_web_store", lambda **kw: None)
    monkeypatch.setattr(run_all_mod, "get_web_jobs", lambda **kw: [])

    run_all_mod.run_all(
        **{**COMMON, "worker": WorkerConfig(
            image_name="user/image", cluster=["ai2/jupiter"], gpus=1
        )}
    )
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
        run_all_mod, "supervise", lambda **kw: seen.append(kw["worker"].env_vars)
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
        run_all_mod, "supervise", lambda **kw: seen.append(kw["worker"].env_vars)
    )
    _stub_paths(monkeypatch, exists=True)
    monkeypatch.setattr(run_all_mod, "get_jobs", lambda **kw: [])

    run_all_mod.run_all(
        **{
            **COMMON,
            "worker": WorkerConfig(
                image_name="user/image",
                cluster=["ai2/jupiter"],
                env_vars={"GS_USER_PROJECT": "other-project"},
            ),
        },
        skip_pca=True,
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

    # Exactly what the CLI hands over. run_all takes **supervise_kwargs, so jsonargparse
    # expands supervise's signature into run_all's own options, and every one of these
    # arrives already set to supervise's default whether asked for or not. A test that
    # omits them cannot see the defect this guards: setdefault does nothing when the key
    # is already present, so the web values never applied on a real run.
    run_all_mod.run_all(
        **COMMON,
        cycle=CycleConfig(seconds=900),
        web_min_zoom=14,
        web_max_zoom=14,
    )

    web = seen[run_all_mod.STAGE_RENDER_WEB_PCA]
    assert web["cycle"].seconds == run_all_mod.WEB_CYCLE_SECONDS
    assert web["cycle"].pending_per_worker == run_all_mod.WEB_PENDING_PER_WORKER
    for stage in ("predict", run_all_mod.STAGE_RENDER_UTM_PCA):
        # They receive whatever the CLI supplied, which for a long-job stage is the
        # right answer: a 15-minute cycle and a shallow queue suit jobs that run for
        # tens of minutes. What matters is that they did not pick up the web values.
        assert seen[stage]["cycle"].seconds == 900
        assert seen[stage]["cycle"].pending_per_worker == 3
        assert seen[stage]["cycle"].seconds != run_all_mod.WEB_CYCLE_SECONDS
        assert seen[stage]["cycle"].pending_per_worker != run_all_mod.WEB_PENDING_PER_WORKER


def test_web_stage_ignores_a_leaked_cycle_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web stage keeps its own timings whatever arrives from the CLI.

    An operator override cannot be honoured here: once jsonargparse has expanded the
    signature, a leaked default and a value someone actually typed are indistinguishable.
    Given the choice between silently ignoring a typed flag and silently ignoring the
    tuning that makes the stage work at all, this takes the first.
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

    run_all_mod.run_all(
        **COMMON,
        web_min_zoom=14,
        web_max_zoom=14,
        cycle=CycleConfig(seconds=45),
    )

    assert (
        seen[run_all_mod.STAGE_RENDER_WEB_PCA]["cycle"].seconds
        == run_all_mod.WEB_CYCLE_SECONDS
    )
    # The long-job stages still take it, which is where an override is meaningful.
    assert seen["predict"]["cycle"].seconds == 45


def test_web_workers_outlive_the_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A web worker must wait longer for work than the supervisor takes to refill.

    The worker default is ten seconds, which is right for a queue filled once up front
    and wrong for one refilled on a cycle: measured on the Kenya rebuild, all 32 workers
    quit seconds after draining the queue, and because a worker counts as live until its
    heartbeat goes stale, the supervisor then waited fifteen minutes before topping up.
    Three minutes of work, fifteen minutes of nothing. The ordering asserted here is what
    stops that pairing coming back.
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

    # Exactly what the CLI hands over. run_all takes **supervise_kwargs, so jsonargparse
    # expands supervise's signature into run_all's own options, and every one of these
    # arrives already set to supervise's default whether asked for or not. A test that
    # omits them cannot see the defect this guards: setdefault does nothing when the key
    # is already present, so the web values never applied on a real run.
    run_all_mod.run_all(
        **COMMON,
        cycle=CycleConfig(seconds=900),
        web_min_zoom=14,
        web_max_zoom=14,
    )

    web = seen[run_all_mod.STAGE_RENDER_WEB_PCA]
    assert web["worker"].idle_seconds == run_all_mod.WEB_WORKER_IDLE_SECONDS
    assert web["worker"].idle_seconds > web["cycle"].seconds
    # The long-job stages keep the worker's own default: a predict worker that idles
    # for fifteen minutes is holding a GPU it is not using.
    for stage in ("predict", run_all_mod.STAGE_RENDER_UTM_PCA):
        assert seen[stage]["worker"].idle_seconds == 900


def test_launch_workers_passes_the_idle_timeout() -> None:
    """supervise's option has to reach the worker's command line to do anything.

    Asserted against the real signature rather than a stub, because the defect this
    guards against is a supervise-side option that is silently accepted and never
    forwarded -- which looks exactly like a working fix until the run is watched.
    """
    import inspect

    from rslp.common.worker import launch_workers

    assert "idle_timeout" in inspect.signature(launch_workers).parameters
    source = inspect.getsource(launch_workers)
    assert '"--idle_timeout"' in source
