"""Tests for supervise's stage dispatch.

The two stages share the shallow-queue and worker-top-up loop; what differs is how
remaining work is enumerated and which workflow the queue entries name. These tests
cover the validation and the enumeration branch without touching Beaker.
"""

import importlib

import pytest

from rslp.large_scale_embeddings.predict_pipeline import EmbeddingInputs

# The package re-exports the workflow functions under their module names, so
# `from rslp.large_scale_embeddings import supervise` yields the function. Import the
# module itself to reach its constants.
sup = importlib.import_module("rslp.large_scale_embeddings.supervise")


def _base_kwargs(**overrides: object) -> dict:
    kwargs = dict(
        inputs=EmbeddingInputs.S2,
        years=[2024],
        store_path="gs://bucket/s2.zarr",
        completed_path_template="gs://bucket/s2_{year}_completed/",
        queue_name="user/queue",
        checkpoint_path="/weka/ckpt",
        image_name="user/image",
        cluster=["ai2/cluster"],
        max_cycles=0,
    )
    kwargs.update(overrides)
    return kwargs


def test_stages_are_the_three_expected() -> None:
    """The stage names are the queue's workflow names, so they are a wire contract."""
    assert sup.STAGES == (
        sup.STAGE_PREDICT,
        sup.STAGE_RENDER_PCA,
        sup.STAGE_REPROJECT_WEB,
    )
    assert sup.STAGE_PREDICT == "predict"
    assert sup.STAGE_RENDER_PCA == "render_pca"
    assert sup.STAGE_REPROJECT_WEB == "reproject_web"


def test_every_stage_has_a_registered_workflow() -> None:
    """A stage a worker cannot run would enqueue jobs that fail one by one."""
    from rslp.large_scale_embeddings import workflows

    for stage in sup.STAGES:
        assert stage in workflows, f"stage {stage} has no worker entry point"


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="stage must be one of"):
        sup.supervise(**_base_kwargs(stage="nonsense"))


def test_render_stage_requires_its_three_paths() -> None:
    with pytest.raises(
        ValueError, match="artifact_path, pca_store_path, pca_completed_path"
    ):
        sup.supervise(**_base_kwargs(stage=sup.STAGE_RENDER_PCA))

    # Naming two of the three still fails, and the error names only what is missing.
    with pytest.raises(ValueError, match="pca_completed_path"):
        sup.supervise(
            **_base_kwargs(
                stage=sup.STAGE_RENDER_PCA,
                artifact_path="gs://bucket/pca",
                pca_store_path="gs://bucket/pca_v1.zarr",
            )
        )

    with pytest.raises(ValueError, match="pca_store_path"):
        sup.supervise(
            **_base_kwargs(
                stage=sup.STAGE_RENDER_PCA,
                artifact_path="gs://bucket/pca",
                pca_completed_path="gs://bucket/pca_completed/",
            )
        )


def test_render_stage_error_points_at_fit_pca() -> None:
    with pytest.raises(ValueError, match="fit_pca"):
        sup.supervise(**_base_kwargs(stage=sup.STAGE_RENDER_PCA))


def test_predict_stage_needs_no_pca_arguments() -> None:
    """max_cycles=0 exits before any Beaker call, so this only exercises validation."""
    sup.supervise(**_base_kwargs())


class _FakeQueueApi:
    """A queue with nothing in it and nobody working it."""

    def get(self, name: str) -> object:
        return object()

    def list_entries(self, queue: object) -> list:
        return []

    def list_workers(self, queue: object) -> list:
        return []


class _FakeBeaker:
    queue = _FakeQueueApi()

    def __enter__(self) -> "_FakeBeaker":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    @classmethod
    def from_env(cls, **kwargs: object) -> "_FakeBeaker":
        return cls()


def _install_cycle_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Stub out Beaker and the enqueue/launch calls, returning what they recorded."""
    recorded: dict[str, object] = {}
    monkeypatch.setattr("beaker.Beaker", _FakeBeaker)

    def fake_write_jobs(
        queue_name: str, project: str, workflow: str, batch: list
    ) -> None:
        recorded["workflow"] = workflow
        recorded["count"] = len(batch)

    def fake_launch_workers(**kwargs: object) -> None:
        recorded["launched"] = kwargs.get("num_workers")
        recorded["gpus"] = kwargs.get("gpus")

    monkeypatch.setattr("rslp.common.worker.write_jobs", fake_write_jobs)
    monkeypatch.setattr("rslp.common.worker.launch_workers", fake_launch_workers)
    return recorded


def _render_cycle_kwargs(**overrides: object) -> dict[str, object]:
    """A complete render-stage kwargs set for _run_cycle, with overrides applied."""
    kwargs: dict[str, object] = {
        "queue_name": "user/queue",
        "num_workers": 2,
        "stale_seconds": 900,
        "years": [2024, 2025],
        "stage": sup.STAGE_RENDER_PCA,
        "store_path": "gs://bucket/s2.zarr",
        "artifact_path": "gs://bucket/pca",
        "pca_store_path": "gs://bucket/pca_v1.zarr",
        "max_level": 3,
        "pca_completed_path": "gs://bucket/pca_completed/",
        "completed_path_template": "gs://bucket/s2_{year}_completed/",
        "patch_size": 1,
        "image_name": "user/image",
        "cluster": ["ai2/cluster"],
        "gpus": 0,
        "shared_memory": "64GiB",
        "priority": "normal",
        "weka_bucket": "b",
        "weka_mount_path": "/m",
        "datasets_api_url": "https://example.invalid",
        "datasets_token_secret": "SECRET",
    }
    kwargs.update(overrides)
    return kwargs


class _Result:
    value = -1


def _stub_render_jobs(monkeypatch: pytest.MonkeyPatch, count: int) -> dict[str, object]:
    """Make the render enumerator return `count` jobs, capturing its arguments."""
    calls: dict[str, object] = {}

    def fake_get_render_jobs(**kwargs: object) -> list[list[str]]:
        calls.update(kwargs)
        return [["--source_marker", f"{i}.json"] for i in range(count)]

    monkeypatch.setattr(
        "rslp.large_scale_embeddings.render_pca.get_render_jobs", fake_get_render_jobs
    )
    return calls


def test_run_cycle_render_stage_enumerates_from_source_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The render stage must call get_render_jobs, not the predict enumerator."""
    calls = _stub_render_jobs(monkeypatch, 2)
    enqueued = _install_cycle_fakes(monkeypatch)
    result = _Result()
    sup._run_cycle(_render_cycle_kwargs(), result)

    # Both years' marker directories are handed to the render enumerator.
    assert calls["source_completed_paths"] == [
        "gs://bucket/s2_2024_completed/",
        "gs://bucket/s2_2025_completed/",
    ]
    assert calls["completed_path"] == "gs://bucket/pca_completed/"
    assert calls["artifact_path"] == "gs://bucket/pca"
    assert calls["pca_store_path"] == "gs://bucket/pca_v1.zarr"
    assert calls["max_level"] == 3
    # Entries name the render workflow, not predict.
    assert result.value == 2
    assert enqueued["workflow"] == sup.STAGE_RENDER_PCA
    assert enqueued["count"] == 2
    # The render stage asks for no GPUs.
    assert enqueued["gpus"] == 0


def test_worker_launches_are_capped_by_outstanding_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tail with fewer jobs than workers must not launch the full worker count.

    A worker holds one job at a time, so a surplus worker starts, finds an empty queue
    and exits, and the next cycle launches it again. On a real resume this churned
    toward num_workers=32 for four outstanding jobs.
    """
    _stub_render_jobs(monkeypatch, 3)
    enqueued = _install_cycle_fakes(monkeypatch)
    result = _Result()
    sup._run_cycle(_render_cycle_kwargs(num_workers=32), result)

    assert result.value == 3
    assert enqueued["launched"] == 3


def test_worker_launches_use_the_full_count_when_work_is_plentiful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap must not throttle the bulk of a run, where jobs outnumber workers."""
    _stub_render_jobs(monkeypatch, 50)
    enqueued = _install_cycle_fakes(monkeypatch)
    result = _Result()
    sup._run_cycle(_render_cycle_kwargs(num_workers=8), result)

    assert result.value == 50
    assert enqueued["launched"] == 8


# ---------------------------------------------------------------- marker detection
#
# A remaining count of zero on the first cycle is ambiguous: either the stage is
# already finished, or the AOI and zone filters excluded everything. Conflating the
# two made a resumed run fail instead of skipping its completed stage, so these
# pin down the distinction.


def test_stage_marker_paths_predict_expands_every_year() -> None:
    paths = sup._stage_marker_paths(
        {
            "stage": sup.STAGE_PREDICT,
            "years": [2023, 2024],
            "completed_path_template": "gs://bucket/done_{year}/",
        }
    )
    assert paths == ["gs://bucket/done_2023/", "gs://bucket/done_2024/"]


def test_stage_marker_paths_render_uses_the_pca_path() -> None:
    paths = sup._stage_marker_paths(
        {
            "stage": sup.STAGE_RENDER_PCA,
            "years": [2024],
            "completed_path_template": "gs://bucket/done_{year}/",
            "pca_completed_path": "gs://bucket/pca_done/",
        }
    )
    assert paths == ["gs://bucket/pca_done/"]


def test_any_completion_markers_true_when_a_marker_exists(tmp_path) -> None:
    done = tmp_path / "done_2024"
    done.mkdir()
    (done / "EPSG:32610_0_0.json").write_text("{}")
    assert sup._any_completion_markers(
        {
            "stage": sup.STAGE_PREDICT,
            "years": [2024],
            "completed_path_template": str(tmp_path / "done_{year}") + "/",
        }
    )


def test_any_completion_markers_false_for_missing_and_empty_dirs(tmp_path) -> None:
    # 2023's directory does not exist at all; 2024's exists but is empty. Neither is
    # evidence of work, so a zero remaining count really does mean nothing matched.
    (tmp_path / "done_2024").mkdir()
    assert not sup._any_completion_markers(
        {
            "stage": sup.STAGE_PREDICT,
            "years": [2023, 2024],
            "completed_path_template": str(tmp_path / "done_{year}") + "/",
        }
    )


# ------------------------------------------------------- repeated cycle failures
#
# A cycle reports nothing whether it was killed for running long or crashed outright.
# Retrying forever is right for the first and wrong for the second: a missing
# geojson_fname failed identically every 15 minutes for an hour on a real run, and the
# job looked alive the whole time.


class _InlineProcess:
    """Runs the cycle in-process instead of spawning.

    supervise uses a spawn context so it can kill a hung cycle, but spawn pickles the
    target, and a test's local function is unpicklable. Running inline keeps the
    supervise loop under test while letting the fake cycle be a closure.
    """

    def __init__(self, target, args):
        self._target, self._args = target, args
        self.exitcode = 0

    def start(self):
        try:
            self._target(*self._args)
        except Exception:  # noqa: BLE001 - mirrors a real cycle crashing for any reason
            self.exitcode = 1

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False

    def terminate(self):
        return None

    def kill(self):
        return None


class _Shared:
    def __init__(self, value):
        self.value = value


class _InlineContext:
    def Value(self, _typecode, init):
        return _Shared(init)

    def Process(self, target, args):
        return _InlineProcess(target, args)


def _inline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sup.multiprocessing, "get_context", lambda _m: _InlineContext())
    monkeypatch.setattr(sup.time, "sleep", lambda _s: None)


def test_failure_cap_is_small_but_tolerates_one_hang() -> None:
    assert 2 <= sup.MAX_CONSECUTIVE_CYCLE_FAILURES <= 5


def test_supervise_raises_after_repeated_cycle_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inline(monkeypatch)
    calls = {"n": 0}

    def never_reports(kwargs, result):
        # Leave result at _NO_RESULT, as a crashed or killed cycle does.
        calls["n"] += 1

    monkeypatch.setattr(sup, "_run_cycle", never_reports)
    with pytest.raises(RuntimeError, match="consecutive cycles failed"):
        sup.supervise(**_base_kwargs(max_cycles=None, cycle_seconds=0))
    assert calls["n"] == sup.MAX_CONSECUTIVE_CYCLE_FAILURES


def test_a_reporting_cycle_resets_the_failure_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # fail, fail, report, fail, fail -> never three consecutively, so no raise; the loop
    # ends on max_cycles instead.
    _inline(monkeypatch)
    script = [None, None, 4, None, None]
    seen = {"i": 0}

    def scripted(kwargs, result):
        val = script[seen["i"] % len(script)]
        seen["i"] += 1
        if val is not None:
            result.value = val

    monkeypatch.setattr(sup, "_run_cycle", scripted)
    sup.supervise(**_base_kwargs(max_cycles=5, cycle_seconds=0))
    assert seen["i"] == 5


# ------------------------------------------------------------ stale claim detection
#
# Claims are never released by the queue, so a dead worker's claim lingers forever.
# Skipping every claimed job would deadlock the run on the first worker death; trusting a
# claim only while it is young keeps recovery while cutting duplicate work.


class _Val:
    def __init__(self, s: str) -> None:
        self.string_value = s


class _ListVal:
    def __init__(self, args: list[str]) -> None:
        self.values = [_Val(a) for a in args]


class _Field:
    def __init__(self, args: list[str]) -> None:
        self.list_value = _ListVal(args)


class _Input:
    def __init__(self, args: list[str]) -> None:
        self.fields = {"args": _Field(args)}


class _Claimed:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds


class _Status:
    def __init__(self, state: str, claimed_at: int = 0) -> None:
        # _state_name reads this; mirror the shape it expects.
        self.state = state
        self.claimed = _Claimed(claimed_at)


class _Entry:
    def __init__(self, args: list[str], state: str, claimed_at: int = 0) -> None:
        self.input = _Input(args)
        self.status = _Status(state, claimed_at)


def test_entry_job_key_reads_the_args() -> None:
    assert sup._entry_job_key(_Entry(["--a", "1"], "PENDING")) == ("--a", "1")


def test_entry_job_key_tolerates_a_malformed_payload() -> None:
    class Broken:
        input = object()

    assert sup._entry_job_key(Broken()) is None


def test_pending_and_fresh_claims_count_as_in_flight(monkeypatch) -> None:
    monkeypatch.setattr(sup, "_state_name", lambda e: e.status.state)
    now = 10_000
    entries = [
        _Entry(["job", "a"], "PENDING"),
        _Entry(["job", "b"], "CLAIMED", claimed_at=now - 60),  # 1 min old
        _Entry(["job", "c"], "CLAIMED", claimed_at=now - 100_000),  # long dead
        _Entry(["job", "d"], "COMPLETED"),
    ]
    keys = sup._in_flight_job_keys(entries, now, claim_stale_seconds=5400)
    assert ("job", "a") in keys, "a pending entry needs no duplicate"
    assert ("job", "b") in keys, "a fresh claim is being worked"
    assert ("job", "c") not in keys, "a stale claim must be re-offered, or work is lost"
    assert ("job", "d") not in keys, "a finished entry says nothing about pending work"


def test_a_claim_without_a_timestamp_is_treated_as_live(monkeypatch) -> None:
    # Re-offering a job that is genuinely being worked costs one duplicate; wrongly
    # skipping one costs the whole run, so the unknown case errs toward live.
    monkeypatch.setattr(sup, "_state_name", lambda e: e.status.state)
    entries = [_Entry(["job", "x"], "CLAIMED", claimed_at=0)]
    assert ("job", "x") in sup._in_flight_job_keys(entries, 10_000)


def test_stale_threshold_is_well_clear_of_one_job() -> None:
    # A predict job at job_size 8192 runs ~25 min; the threshold must not sit near that.
    assert sup.DEFAULT_CLAIM_STALE_SECONDS >= 3 * 25 * 60
