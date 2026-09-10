import pytest


def test_every_supervise_option_reaches_the_cycle() -> None:
    """Each `supervise` parameter must be forwarded into the config the cycle reads.

    supervise runs each cycle in a spawned process and hands it one `SuperviseConfig`.
    An option accepted by the signature but never put into that object is accepted,
    documented, and silently ignored: `worker_idle_seconds` shipped that way, and every
    worker still used the ten-second default while the run looked correct. Comparing
    the signature against the construction catches the whole class of that bug.
    """
    import ast
    import importlib
    import inspect

    # importlib, not `from ... import supervise`: the package re-exports the function
    # of that name, so a plain import would hand getsource one function body instead of
    # the module.
    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    assert inspect.ismodule(mod), "expected the module, got the re-exported function"
    tree = ast.parse(inspect.getsource(mod))

    forwarded: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SuperviseConfig"
        ):
            forwarded.update(kw.arg for kw in node.keywords if kw.arg)
    assert forwarded, "could not find the SuperviseConfig construction"

    params = set(inspect.signature(mod.supervise).parameters) - {"self"}
    missing = params - forwarded
    assert not missing, (
        f"supervise accepts {sorted(missing)} but never puts them in SuperviseConfig, "
        "so the cycle cannot see them and the options are silently ignored"
    )


def test_the_config_objects_carry_every_field_the_cycle_reads() -> None:
    """`_run_cycle` must read only attributes the config dataclasses actually define.

    The config replaced a `dict[str, Any]`, where a typo read as None forever. Keeping
    the reads checked against the dataclass fields is what makes that impossible.
    """
    import dataclasses
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    for cls in (
        mod.SuperviseConfig,
        mod.ModelConfig,
        mod.WorkerConfig,
        mod.CycleConfig,
        mod.AoiConfig,
        mod.PcaConfig,
    ):
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} must stay a dataclass"


def test_the_worker_count_is_not_time_based() -> None:
    """The pool size must be derived from state, not from a startup timer.

    Counting workers by queue registration misses every worker still starting, so the
    shortfall gets launched again each cycle and the pool overshoots `num_workers` by
    however many cycles a container start takes. A timer covering that window only
    narrows the race: set too short it overshoots anyway, set too long it leaves the
    pool short whenever a worker dies while starting. `_count_workers` asks Beaker
    which worker experiments exist and have not finalized, which is exact.
    """
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    params = inspect.signature(mod.supervise).parameters
    for name in ("worker_startup_seconds", "stale_seconds"):
        assert name not in params, (
            f"{name} is back: the worker count is a timer again, so the pool will "
            "overshoot num_workers whenever a container start outlasts it"
        )


def test_workers_are_named_so_a_run_can_count_its_own() -> None:
    """Worker names must distinguish one run's workers from another's.

    The count is a name-prefix match over unfinalized experiments, so a bare
    `worker_<random>` name would make every concurrent run's workers count toward every
    other run's target.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    mine = mod.worker_name_prefix("user/queue-a")
    theirs = mod.worker_name_prefix("user/queue-b")
    assert mine != theirs
    assert not mine.startswith(theirs) and not theirs.startswith(
        mine
    ), "one queue's prefix matches another's, so their worker counts would collide"
    assert "/" not in mine, "a Beaker experiment name cannot contain a slash"


def test_a_worker_outlasts_the_gap_between_refills() -> None:
    """A worker must survive until the next cycle can hand it work.

    A cycle re-enumerates every tile before it sleeps, so the real interval between
    refills is that work plus `CycleConfig.seconds`, bounded above by `budget_seconds`
    because the parent kills a cycle that overruns. A worker that idles out sooner has
    to be relaunched and pay container start again.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    cycle = mod.CycleConfig()
    idle = mod.WorkerConfig(image_name="i", cluster=["c"]).idle_seconds
    assert idle is not None, (
        "WorkerConfig.idle_seconds defaults to None, which hands the worker its own "
        "ten-second timeout, so it quits the moment the queue drains"
    )
    worst_case = cycle.budget_seconds + cycle.seconds
    assert idle >= worst_case, (
        f"a worker idles out after {idle}s but a cycle can take up to {worst_case}s "
        "to come back round, so the pool empties between refills"
    )


def test_workers_always_get_the_gdal_billing_project() -> None:
    """A WorkerConfig must carry the GDAL env whether or not the caller passed any.

    The requester-pays USGS Landsat mirror needs GDAL handed a billing project via
    GS_USER_PROJECT; without it every Landsat read fails with an HTTP 400 that names no
    variable, and the run carries on producing embeddings with the Landsat inputs
    missing. Silent degradation, not a crash.

    The merge used to live in one caller, so a supervisor launched through another
    route skipped it entirely and five diagnostic runs went out that way. Doing it in
    __post_init__ makes it a property of the config rather than of the caller.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    assert "GS_USER_PROJECT" in mod.DEFAULT_WORKER_ENV_VARS

    envs: list[dict[str, str] | None] = [None, {}, {"SOMETHING_ELSE": "1"}]
    for env in envs:
        worker = mod.WorkerConfig(image_name="i", cluster=["c"], env_vars=env)
        assert worker.env_vars["GS_USER_PROJECT"], (
            f"env_vars={env!r} produced a WorkerConfig with no billing project, so "
            "its workers would read Landsat unauthorised"
        )


def test_an_explicit_worker_env_var_still_wins() -> None:
    """Merging defaults must not stop a caller overriding one.

    The merge order matters: defaults first, caller second. Reversed, a run could not
    point at a different billing project.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = mod.WorkerConfig(
        image_name="i", cluster=["c"], env_vars={"GS_USER_PROJECT": "other-project"}
    )
    assert worker.env_vars["GS_USER_PROJECT"] == "other-project"


class _FakeSeconds:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds


class _FakeExperiment:
    def __init__(self, name: str, created: int) -> None:
        self.name = name
        self.created = _FakeSeconds(created)


class _FakeWorkload:
    def __init__(self, name: str, created: int) -> None:
        self.experiment = _FakeExperiment(name, created)


class _FakeHeartbeat:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds


class _FakeQueueWorker:
    def __init__(self, seconds: int) -> None:
        self.heartbeat = _FakeHeartbeat(seconds)


class _FakeBeaker:
    """Just enough of the client for _count_workers."""

    def __init__(self, workloads: list, heartbeats: list[int]) -> None:
        self._workloads = workloads
        self._heartbeats = heartbeats
        outer = self

        class _FakeJobStatus:
            def HasField(self, name: str) -> bool:  # noqa: N802 - protobuf API
                # Every workload here has started. Counting by started-ness alone is
                # what left the overshoot hole, so the fake has to model it.
                return name == "started"

        class _FakeJob:
            status = _FakeJobStatus()

        class _WorkloadSvc:
            def list(self, **kwargs: object) -> list:
                return outer._workloads

            def get_latest_job(self, workload: object) -> object:
                return _FakeJob()

        class _QueueSvc:
            def list_workers(self, queue: object) -> list:
                return [_FakeQueueWorker(s) for s in outer._heartbeats]

        class _UserSvc:
            def get(self) -> str:
                return "me"

        self.workload = _WorkloadSvc()
        self.queue = _QueueSvc()
        self.user = _UserSvc()


def test_a_worker_that_stopped_heartbeating_does_not_count() -> None:
    """A dead worker must not hold a slot, or the run strands itself.

    Launches are capped by outstanding work. A worker whose process died stays
    unfinalized to Beaker forever, so counting it means that once the dead count
    reaches the outstanding count the supervisor launches nothing and the last jobs are
    never claimed. That is exactly how the web pyramid deadlocked at a 16-shard zoom.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    now = 1_000_000.0
    # Twenty workloads Beaker still calls live, none of them heartbeating.
    old = int(now) - 7200  # created two hours ago, well past the startup grace
    workloads = [_FakeWorkload(f"{prefix}_{i}", old) for i in range(20)]
    beaker = _FakeBeaker(workloads, heartbeats=[])
    assert (
        mod._count_workers(beaker, object(), prefix, queue=object(), now=now) == 0
    ), "dead workers still count, so the pool will strand the last jobs"


def test_a_starting_worker_still_counts() -> None:
    """Container start takes minutes; without this the pool overshoots every cycle."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    now = 1_000_000.0
    workloads = [_FakeWorkload(f"{prefix}_{i}", int(now) - 60) for i in range(5)]
    beaker = _FakeBeaker(workloads, heartbeats=[])
    assert mod._count_workers(beaker, object(), prefix, queue=object(), now=now) == 5


def test_a_busy_worker_counts_via_its_heartbeat() -> None:
    """The SDK refreshes the registration from a background thread.

    So a worker stays fresh through a job that runs for tens of minutes, and must keep
    its slot rather than being relaunched alongside itself.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    now = 1_000_000.0
    workloads = [_FakeWorkload(f"{prefix}_{i}", int(now) - 7200) for i in range(3)]
    beaker = _FakeBeaker(workloads, heartbeats=[int(now) - 10] * 3)
    assert mod._count_workers(beaker, object(), prefix, queue=object(), now=now) == 3


def test_a_started_worker_counts_before_it_registers() -> None:
    """The overshoot hole: started, but not yet on the queue.

    A worker registers only after importing torch and the rslp stack, so there is a
    window of tens of seconds where it is running and heartbeating nothing. Treating
    that as absent relaunches it, and a fresh 128-worker pool overshoots by however
    many are inside the window when the cycle samples.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    now = 1_000_000.0
    # Created a minute ago: image pulled, container up, registration not in yet.
    workloads = [_FakeWorkload(f"{prefix}_{i}", int(now) - 60) for i in range(128)]
    beaker = _FakeBeaker(workloads, heartbeats=[])
    assert (
        mod._count_workers(beaker, object(), prefix, queue=object(), now=now) == 128
    ), "a starting worker reads as absent, so the pool will overshoot num_workers"


def test_metrics_are_off_by_default() -> None:
    """No project configured must mean no wandb import and no network call."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    assert mod.CycleConfig().wandb_project is None
    m = mod._Metrics(None, "run", {})
    m.log(1, {"jobs/remaining": 5})
    m.finish()


def test_a_broken_metrics_sink_cannot_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A month-long run must not die because a metrics backend is unreachable.

    An expired key, an outage or a missing package all arrive as an exception from
    wandb. Each one has to degrade to not logging, because the supervisor is the only
    thing keeping the run alive.
    """
    import importlib
    import sys
    import types

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    broken = types.ModuleType("wandb")

    def explode(**kwargs: object) -> object:
        raise RuntimeError("no network")

    broken.init = explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", broken)
    m = mod._Metrics("some-project", "run", {"a": 1})
    assert m._run is None, "a failed init must leave an inert sink"
    m.log(1, {"jobs/remaining": 5})  # must not raise
    m.finish()

    # A sink that initialises but then fails on log must also go inert, not raise.
    class _Run:
        def log(self, *a: object, **k: object) -> None:
            raise RuntimeError("dropped")

        def finish(self) -> None:
            raise RuntimeError("dropped")

    working = types.ModuleType("wandb")
    working.init = lambda **kwargs: _Run()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", working)
    m2 = mod._Metrics("some-project", "run", {})
    assert m2._run is not None
    m2.log(1, {"jobs/remaining": 5})
    assert m2._run is None, "a failed log must disable further logging"
    m2.finish()


def test_unmeasured_metrics_are_dropped_not_zeroed() -> None:
    """A killed cycle has no job count; logging 0 would draw a false cliff."""
    import importlib
    import sys
    import types

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    sent: list[dict] = []

    class _Run:
        def log(self, values: dict, step: int | None = None) -> None:
            sent.append(values)

        def finish(self) -> None:
            pass

    fake = types.ModuleType("wandb")
    fake.init = lambda **kwargs: _Run()  # type: ignore[attr-defined]
    sys.modules["wandb"] = fake
    try:
        m = mod._Metrics("p", "r", {})
        m.log(1, {"jobs/remaining": None, "queue/pending": 12})
    finally:
        del sys.modules["wandb"]
    assert sent == [{"queue/pending": 12}]
