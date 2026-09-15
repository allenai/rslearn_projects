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
    def __init__(self, name: str, created: int, status: int = 0) -> None:
        self.experiment = _FakeExperiment(name, created)
        # 4 == running in Beaker's workload status enum; 0 stands in for "not started".
        self.status = status


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
    workloads = [
        _FakeWorkload(f"{prefix}_{i}", int(now) - 60, status=3) for i in range(5)
    ]
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


def test_a_long_queued_worker_still_counts() -> None:
    """A saturated cluster leaves workers queued for hours; they must keep counting.

    Aging a queued worker out of the count makes the supervisor believe it is short and
    launch another, which also queues. Measured on Jupiter while scaling to 256: queued
    workers waited a median of 118 minutes, 221 of 240 were past any sane startup
    grace, and the supervisor had put 532 workloads on the cluster against a target of
    256 before anyone looked. Queued workers hold no GPU, so throughput looked fine
    throughout.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    now = 1_000_000.0
    # Queued for five hours, far past any startup window.
    workloads = [
        _FakeWorkload(f"{prefix}_{i}", int(now) - 18_000, status=2) for i in range(240)
    ]
    beaker = _FakeBeaker(workloads, heartbeats=[])
    assert (
        mod._count_workers(beaker, object(), prefix, queue=object(), now=now) == 240
    ), "a long-queued worker stopped counting, so the supervisor will launch more"


def test_a_running_worker_is_not_counted_twice() -> None:
    """Registration follows container start by seconds, so both signals see it.

    Counting a young *running* worker as "starting" as well as counting its heartbeat
    inflates the pool by a whole launch batch: a measured scale-up to 256 reported 384
    workers against 291 real ones, and a supervisor that believes it is over target
    will not backfill until the batch ages out of the startup window.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    now = 1_000_000.0
    # 128 workers launched a minute ago, all already running and all heartbeating.
    workloads = [
        _FakeWorkload(f"{prefix}_{i}", int(now) - 60, status=4) for i in range(128)
    ]
    beaker = _FakeBeaker(workloads, heartbeats=[int(now) - 10] * 128)
    assert (
        mod._count_workers(beaker, object(), prefix, queue=object(), now=now) == 128
    ), "a young running worker is counted by both signals, so the pool is overstated"


def test_a_queued_worker_still_counts_while_it_starts() -> None:
    """Not yet running means not yet registered, so nothing else is counting it."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    now = 1_000_000.0
    # status 2 == queued: scheduled but not executing, so no heartbeat exists yet.
    workloads = [
        _FakeWorkload(f"{prefix}_{i}", int(now) - 60, status=2) for i in range(10)
    ]
    beaker = _FakeBeaker(workloads, heartbeats=[])
    assert mod._count_workers(beaker, object(), prefix, queue=object(), now=now) == 10


class _FakeJob:
    def __init__(self, workspace_id: str, gpus: int) -> None:
        self.workspace_id = workspace_id

        class _RR:
            gpu_count = gpus

        class _CS:
            resource_request = _RR()

        self.container_spec = _CS()


class _FakeWs:
    def __init__(self, wid: str) -> None:
        self.id = wid


class _FakeCapacityBeaker:
    """Just enough of the client for _capacity_target under allocation sizing."""

    def __init__(self, jobs: list | Exception, ws_id: str = "WS") -> None:
        outer = self
        self.cluster_calls: list[str] = []
        self.job_list_kwargs: list[dict] = []

        class _ClusterSvc:
            def get(self, name: str, include_cluster_occupancy: bool = False) -> object:
                outer.cluster_calls.append(name)
                return object()

        class _JobSvc:
            def list(self, **kw: object) -> list:
                outer.job_list_kwargs.append(dict(kw))
                if isinstance(jobs, Exception):
                    raise jobs
                return jobs

        class _WsSvc:
            def get(self, name: str) -> _FakeWs:
                return _FakeWs(ws_id)

        self.cluster = _ClusterSvc()
        self.job = _JobSvc()
        self.workspace = _WsSvc()


def _worker_cfg(mod, **kw):  # type: ignore[no-untyped-def]
    base = dict(
        image_name="i",
        cluster=["ai2/jupiter"],
        num_workers=512,
        gpus=1,
        capacity_slots=352,
    )
    base.update(kw)
    return mod.WorkerConfig(**base)


def test_a_cpu_stage_is_never_sized_from_the_gpu_allocation() -> None:
    """A stage that requests no GPU occupies none of the allocation.

    Sizing it against GPU slots would size it against a number it does not move.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = _worker_cfg(mod, gpus=0, num_workers=128, capacity_fraction=0.75)
    beaker = _FakeCapacityBeaker(jobs=[])
    assert mod._capacity_target(beaker, worker, live=128) == 128
    assert beaker.cluster_calls == [], "allocation was read for a stage with no GPUs"


def test_other_teams_on_the_cluster_do_not_shrink_our_allocation() -> None:
    """The allocation is an entitlement, not a share of whatever is left over.

    Another org saturating jupiter neither grants nor removes what earth-systems may
    use, so their jobs must not appear in the arithmetic at all. Sizing against
    cluster-wide free slots would forfeit capacity the run is owed.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = _worker_cfg(mod, capacity_fraction=0.75, num_workers=512)
    # 900 GPUs held by a different workspace; ours holds nothing.
    foreign = [_FakeJob("SOMEONE_ELSE", 1) for _ in range(900)]
    got = mod._capacity_target(_FakeCapacityBeaker(jobs=foreign), worker, live=232)
    assert got == 264, f"another org's usage changed our target, got {got} (want 264)"


def test_the_fraction_of_the_allocation_is_the_ceiling() -> None:
    """0.75 of 352 is 264, and an idle workspace does not get more than that."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = _worker_cfg(mod, capacity_fraction=0.75, num_workers=512)
    got = mod._capacity_target(_FakeCapacityBeaker(jobs=[]), worker, live=240)
    assert got == 264, f"expected the 0.75 x 352 ceiling, got {got}"


def test_colleagues_in_the_same_workspace_reduce_what_is_left() -> None:
    """The allocation is shared, so their usage is the tighter of the two bounds."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = _worker_cfg(mod, capacity_fraction=0.75, num_workers=512)
    # 200 held by colleagues plus 200 of ours. Ours is subtracted back out, so others
    # are 200 and the remainder is 152, tighter than the 264 ceiling. live is high
    # enough that the +32 growth cap does not mask the difference: ignoring colleagues
    # would give 264, which the cap would show as 232.
    jobs = [_FakeJob("WS", 1) for _ in range(400)]
    got = mod._capacity_target(_FakeCapacityBeaker(jobs=jobs), worker, live=200)
    assert got == 152, f"colleagues' usage was ignored, got {got} (want 352-200)"


def test_our_own_pool_does_not_count_against_our_headroom() -> None:
    """Otherwise scaling up shrinks the number the target is computed from.

    Our workers sit in the same workspace as everyone else's, so without subtracting
    them the pool ratchets itself toward zero as it grows.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = _worker_cfg(mod, capacity_fraction=0.75, num_workers=512)
    ours = [_FakeJob("WS", 1) for _ in range(264)]
    got = mod._capacity_target(_FakeCapacityBeaker(jobs=ours), worker, live=264)
    assert got == 264, f"our own pool ate our headroom, got {got}"


def test_capacity_needs_to_know_how_big_the_allocation_is() -> None:
    """Sizing against an allocation without its size would be sizing against nothing."""
    import importlib

    import pytest as _pytest

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = _worker_cfg(mod, capacity_fraction=0.75, capacity_slots=None)
    with _pytest.raises(ValueError, match="capacity_slots"):
        mod._capacity_target(_FakeCapacityBeaker(jobs=[]), worker, live=0)


def test_capacity_holds_the_pool_when_usage_cannot_be_read() -> None:
    """Guessing could dump a launch onto an allocation colleagues are using."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = _worker_cfg(mod, capacity_fraction=0.75, num_workers=512)
    beaker = _FakeCapacityBeaker(jobs=RuntimeError("beaker is down"))
    assert mod._capacity_target(beaker, worker, live=100) == 100


def test_a_static_pool_is_untouched_by_default() -> None:
    """capacity_fraction unset must behave exactly as before."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = _worker_cfg(mod, num_workers=128)
    beaker = _FakeCapacityBeaker(jobs=[])
    assert mod._capacity_target(beaker, worker, live=4) == 128
    assert beaker.cluster_calls == [], "allocation was read for a static pool"


def test_capacity_converts_slots_to_workers_for_a_multi_gpu_stage() -> None:
    """The allocation is in GPU slots; the target is a worker count.

    They coincide only while a worker holds one GPU. Without the conversion a stage
    asking for 4 GPUs would launch four times the pool the allocation permits.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    # 0.75 * 352 = 264 slots at 4 GPUs each is 66 workers. live is high enough that the
    # growth cap does not mask the difference.
    worker = _worker_cfg(mod, capacity_fraction=0.75, gpus=4, num_workers=1024)
    got = mod._capacity_target(_FakeCapacityBeaker(jobs=[]), worker, live=60)
    assert got == 66, f"slots were treated as workers for a 4-GPU stage, got {got}"


class _ReleaseBeaker:
    """Just enough of the client for _release_surplus_workers."""

    def __init__(self, workloads: list, fail: bool = False) -> None:
        outer = self
        self.cancelled: list = []

        class _WorkloadSvc:
            def list(self, **kwargs: object) -> list:
                return workloads

            # Annotated None, not list: inside this class body the name `list` is
            # the method above, not the builtin.
            def cancel(self, *w: object) -> None:
                if fail:
                    raise RuntimeError("beaker said no")
                outer.cancelled.extend(w)

        class _UserSvc:
            def get(self) -> str:
                return "me"

        self.workload = _WorkloadSvc()
        self.user = _UserSvc()


def test_a_running_worker_is_never_cancelled_to_free_capacity() -> None:
    """A running worker owns a claimed job and there is no intra-job checkpointing.

    Killing one throws away up to a whole unit of work, and the claim it leaves behind
    is not released for 90 minutes. Freeing capacity must never cost work: only workers
    that have not started yet are fair game.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    running = [_FakeWorkload(f"{prefix}_r{i}", 1000 + i, status=4) for i in range(10)]
    beaker = _ReleaseBeaker(running)
    freed = mod._release_surplus_workers(beaker, object(), prefix, surplus=6)
    assert freed == 0, "running workers were cancelled, losing claimed work"
    assert beaker.cancelled == [], "a running worker was cancelled"


def test_surplus_queued_workers_are_released() -> None:
    """Declining to launch more is not enough; the surplus has to be given back."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    waiting = [_FakeWorkload(f"{prefix}_w{i}", 1000 + i, status=2) for i in range(10)]
    beaker = _ReleaseBeaker(waiting)
    assert mod._release_surplus_workers(beaker, object(), prefix, surplus=4) == 4
    assert len(beaker.cancelled) == 4


def test_the_newest_queued_workers_are_released_first() -> None:
    """The tail of a launch burst is the surplus; the oldest are about to start."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    waiting = [_FakeWorkload(f"{prefix}_w{i}", 1000 + i, status=2) for i in range(6)]
    beaker = _ReleaseBeaker(waiting)
    mod._release_surplus_workers(beaker, object(), prefix, surplus=2)
    names = sorted(w.experiment.name for w in beaker.cancelled)
    assert names == [
        f"{prefix}_w4",
        f"{prefix}_w5",
    ], f"released the wrong ones: {names}"


def test_releasing_only_touches_this_runs_workers() -> None:
    """The workspace is shared. Cancelling someone else's job would be unforgivable."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    others = [
        _FakeWorkload(f"worker_someone-else_{i}", 2000 + i, status=2) for i in range(5)
    ]
    mine = [_FakeWorkload(f"{prefix}_w{i}", 1000 + i, status=2) for i in range(2)]
    beaker = _ReleaseBeaker(others + mine)
    mod._release_surplus_workers(beaker, object(), prefix, surplus=5)
    names = {w.experiment.name for w in beaker.cancelled}
    assert names == {f"{prefix}_w0", f"{prefix}_w1"}, f"cancelled foreign work: {names}"


def test_a_failed_cancel_is_not_fatal() -> None:
    """The pool being over target is not an emergency; the next cycle tries again."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    waiting = [_FakeWorkload(f"{prefix}_w{i}", 1000 + i, status=2) for i in range(4)]
    assert (
        mod._release_surplus_workers(
            _ReleaseBeaker(waiting, fail=True), object(), prefix, 3
        )
        == 0
    )


def test_nothing_is_released_when_the_pool_is_within_target() -> None:
    """No surplus, no cancellation."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    waiting = [_FakeWorkload(f"{prefix}_w{i}", 1000 + i, status=2) for i in range(4)]
    beaker = _ReleaseBeaker(waiting)
    assert mod._release_surplus_workers(beaker, object(), prefix, surplus=0) == 0
    assert beaker.cancelled == []


def test_queued_allocation_requests_count_against_the_ceiling() -> None:
    """A colleague's queued job is a claim on the allocation, not free capacity.

    It has not been placed on a node yet, so a scheduled-only filter misses it, and
    this pool would take slots someone is already waiting for. Eligibility is the
    looser predicate and can overcount a job that lists several clusters, which is the
    safe direction for a ceiling we are trying not to exceed.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = _worker_cfg(mod, capacity_fraction=0.75, num_workers=512)
    beaker = _FakeCapacityBeaker(jobs=[])
    mod._capacity_target(beaker, worker, live=0)

    assert beaker.job_list_kwargs, "no job listing was made"
    kw = beaker.job_list_kwargs[0]
    assert (
        "elegible_for_cluster" in kw
    ), f"queued requests are not counted; filter was {sorted(kw)}"
    assert (
        "scheduled" not in kw
    ), "a scheduled-only filter excludes queued claims on the allocation"
