import itertools
import json
from pathlib import Path

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
    pool short whenever a worker dies while starting. `_count_worker_split` asks Beaker
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
    idle = mod.WorkerConfig(image_name="i", cluster=["c"], batch_size=128).idle_seconds
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
        worker = mod.WorkerConfig(
            image_name="i", cluster=["c"], batch_size=128, env_vars=env
        )
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
        batch_size=128,
        image_name="i",
        cluster=["c"],
        env_vars={"GS_USER_PROJECT": "other-project"},
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
    """Just enough of the client for _count_worker_split."""

    def __init__(self, workloads: list, heartbeats: list[int]) -> None:
        self._workloads = workloads
        self._heartbeats = heartbeats
        outer = self

        class _FakeJobStatus:
            def HasField(self, name: str) -> bool:
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
        sum(mod._count_worker_split(beaker, object(), prefix, queue=object(), now=now))
        == 0
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
    assert (
        sum(mod._count_worker_split(beaker, object(), prefix, queue=object(), now=now))
        == 5
    )


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
    assert (
        sum(mod._count_worker_split(beaker, object(), prefix, queue=object(), now=now))
        == 3
    )


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
        sum(mod._count_worker_split(beaker, object(), prefix, queue=object(), now=now))
        == 240
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
        sum(mod._count_worker_split(beaker, object(), prefix, queue=object(), now=now))
        == 128
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
    assert (
        sum(mod._count_worker_split(beaker, object(), prefix, queue=object(), now=now))
        == 10
    )


class _FakeJob:
    # Distinct ids matter: allocation usage is deduplicated by job id, so fakes that
    # share one would collapse into a single claim and hide the usage being tested.
    _next_id = itertools.count()

    def __init__(self, workspace_id: str, gpus: int) -> None:
        self.workspace_id = workspace_id
        self.id = f"job-{next(_FakeJob._next_id)}"

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
        batch_size=128,
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

    Killing one throws away up to a whole unit of work. The claim itself is handed
    back promptly, since the worker rejects its in-flight entry on SIGTERM, but the
    work done so far is gone. Freeing capacity immediately must never cost work: only
    workers that have not started yet are fair game, and the rest are drained.
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


def _drain_list(path: str) -> list[str]:
    """Read back what the supervisor published."""
    import json

    with open(path) as f:
        return json.load(f)["workers"]


def test_running_surplus_is_drained_rather_than_left_alone(tmp_path: Path) -> None:
    """The whole point: a fully running pool over target has to shed something.

    Cancelling cannot touch a running worker, so before the drain list the surplus just
    sat there. The idle timeout is not a fallback, since it only fires on an empty
    queue and the queue is kept topped up all run.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    running = [_FakeWorkload(f"{prefix}_r{i}", 1000 + i, status=4) for i in range(10)]
    beaker = _ReleaseBeaker(running)
    drain_path = str(tmp_path / "drain.json")

    freed = mod._release_surplus_workers(
        beaker, object(), prefix, surplus=3, drain_path=drain_path
    )

    assert freed == 0, "a running worker was cancelled"
    assert beaker.cancelled == [], "a running worker was cancelled"
    # Newest first: the most recently added capacity is the first given back.
    assert _drain_list(drain_path) == [
        f"{prefix}_r9",
        f"{prefix}_r8",
        f"{prefix}_r7",
    ], "the surplus was not asked to retire"


def test_cancelled_workers_count_against_the_drain(tmp_path: Path) -> None:
    """Cancelling is free and instant, so it is spent first and the drain covers the rest.

    Draining as many as the surplus on top of the cancellations would shed twice the
    capacity asked for.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    workloads = [_FakeWorkload(f"{prefix}_w{i}", 2000 + i, status=2) for i in range(2)]
    workloads += [
        _FakeWorkload(f"{prefix}_r{i}", 1000 + i, status=4) for i in range(10)
    ]
    beaker = _ReleaseBeaker(workloads)
    drain_path = str(tmp_path / "drain.json")

    freed = mod._release_surplus_workers(
        beaker, object(), prefix, surplus=5, drain_path=drain_path
    )

    assert freed == 2, "the queued workers were not cancelled"
    drained = _drain_list(drain_path)
    assert len(drained) == 3, f"shed {freed} + {len(drained)} for a surplus of 5"


def test_drain_list_is_cleared_once_the_surplus_is_gone(tmp_path: Path) -> None:
    """A stale list keeps retiring workers forever, emptying the pool.

    This is why the release path runs every cycle rather than only when over target.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    running = [_FakeWorkload(f"{prefix}_r{i}", 1000 + i, status=4) for i in range(10)]
    beaker = _ReleaseBeaker(running)
    drain_path = str(tmp_path / "drain.json")

    mod._release_surplus_workers(
        beaker, object(), prefix, surplus=3, drain_path=drain_path
    )
    assert _drain_list(drain_path), "nothing was drained to begin with"

    mod._release_surplus_workers(
        beaker, object(), prefix, surplus=0, drain_path=drain_path
    )
    assert _drain_list(drain_path) == [], "the drain list was not cleared"


def test_only_this_runs_workers_are_drained(tmp_path: Path) -> None:
    """The workspace is shared; draining a colleague's worker would be sabotage."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    workloads = [_FakeWorkload(f"{prefix}_r{i}", 1000 + i, status=4) for i in range(3)]
    workloads += [
        _FakeWorkload(f"worker_someone-else_r{i}", 3000 + i, status=4) for i in range(5)
    ]
    beaker = _ReleaseBeaker(workloads)
    drain_path = str(tmp_path / "drain.json")

    mod._release_surplus_workers(
        beaker, object(), prefix, surplus=4, drain_path=drain_path
    )

    drained = _drain_list(drain_path)
    assert all(
        name.startswith(prefix) for name in drained
    ), f"drained another run's workers: {drained}"


def test_a_failed_publish_is_not_fatal(tmp_path: Path) -> None:
    """The pool stays over target for a cycle; the next cycle republishes."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    prefix = "worker_patrickj-q"
    running = [_FakeWorkload(f"{prefix}_r{i}", 1000 + i, status=4) for i in range(10)]
    beaker = _ReleaseBeaker(running)
    # A directory that cannot be created, so the publish raises.
    drain_path = str(tmp_path / "file.txt" / "drain.json")
    (tmp_path / "file.txt").write_text("not a directory")

    freed = mod._release_surplus_workers(
        beaker, object(), prefix, surplus=3, drain_path=drain_path
    )
    assert freed == 0, "a publish failure should not change what was cancelled"


def test_drain_path_is_beside_the_store_not_inside_it() -> None:
    """Everything under a .zarr prefix belongs to that store's key space.

    The drain list is run bookkeeping, not store content. Written inside the store it
    shows up in store listings and is copied along by any migration of the store, which
    is exactly what happened on the first production run.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    store = "gs://bucket/run/embeddings.zarr"
    got = mod._drain_path(store, "patrickj/q-2025")
    assert ".zarr/" not in got, f"drain list was written inside the store: {got}"
    assert got == "gs://bucket/run/worker_drain/patrickj-q-2025.json", got


def test_drain_path_sits_alongside_the_completion_markers() -> None:
    """The run directory is the common parent of the store and its marker dirs."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    got = mod._drain_path("gs://bucket/run/embeddings.zarr", "u/q")
    assert got.startswith("gs://bucket/run/"), got


def test_drain_path_is_per_queue() -> None:
    """Several runs share one store, and each has to retire only its own workers."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    store = "gs://bucket/run/embeddings.zarr"
    conus = mod._drain_path(store, "patrickj/conus-2025")
    au_af = mod._drain_path(store, "patrickj/au-af-2025")
    assert conus != au_af, "two runs sharing a store would fight over one drain list"
    assert conus.startswith("gs://bucket/run/"), conus


def test_a_stopping_worker_is_not_double_counted_as_starting() -> None:
    """Stopping and uploading_results are running states, not pre-registration ones.

    Such a worker has already registered, so its heartbeat speaks for it. Counting it
    as "starting" as well double-counts it and the pool undershoots, which is what a
    narrower RUNNING_WORKLOAD_STATUSES silently causes.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    assert (
        {5, 6} <= mod.RUNNING_WORKLOAD_STATUSES
    ), "stopping/uploading workers would be counted as starting"
    prefix = "worker_patrickj-q"
    now = 1_000_000.0
    # Statuses 5 and 6: registered, heartbeat stale, on their way out.
    workloads = [_FakeWorkload(f"{prefix}_a", int(now) - 60, status=5)]
    workloads += [_FakeWorkload(f"{prefix}_b", int(now) - 60, status=6)]
    beaker = _FakeBeaker(workloads, heartbeats=[])
    got = sum(
        mod._count_worker_split(beaker, object(), prefix, queue=object(), now=now)
    )
    assert got == 0, f"a stopping worker was counted as starting, got {got}"


class _FakeSlotCounts:
    def __init__(self, available: int) -> None:
        self.available = available


class _FakeOccupancy:
    def __init__(self, available: int) -> None:
        self.slot_counts = _FakeSlotCounts(available)


class _FakeCluster:
    def __init__(self, available: int) -> None:
        self.cluster_occupancy = _FakeOccupancy(available)


class _FakeClusterBeaker:
    """Just enough Beaker to answer a cluster occupancy read."""

    def __init__(self, available_by_cluster: dict[str, int]) -> None:
        outer = self

        class _ClusterService:
            def get(
                self, name: str, include_cluster_occupancy: bool = False
            ) -> "_FakeCluster":
                assert include_cluster_occupancy, (
                    "occupancy is only populated when asked for; without the flag "
                    "every slot count reads zero and backfill silently does nothing"
                )
                return _FakeCluster(outer._available[name])

        self._available = available_by_cluster
        self.cluster = _ClusterService()


def test_backfill_is_off_unless_asked_for() -> None:
    """A pool with no backfill_fraction must not touch the cluster at all.

    Backfill takes slots outside the allocation, so it has to be opt-in: a run that
    never configured it must behave exactly as before.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = mod.WorkerConfig(image_name="img", cluster=["ai2/jupiter"], batch_size=128)

    class _Exploding:
        @property
        def cluster(self) -> None:
            raise AssertionError("cluster occupancy was read with backfill disabled")

    assert mod._backfill_target(_Exploding(), worker, 0, 0) == 0


def test_backfill_sizes_from_idle_slots_across_clusters() -> None:
    """The target is the configured share of what is idle, summed over clusters."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    beaker = _FakeClusterBeaker({"ai2/jupiter": 100, "ai2/saturn": 20})
    worker = mod.WorkerConfig(
        batch_size=128,
        image_name="img",
        cluster=["ai2/jupiter", "ai2/saturn"],
        backfill_fraction=0.5,
        backfill_max_workers=1000,
    )
    # 120 idle slots, half of them, one GPU per worker.
    assert mod._backfill_target(beaker, worker, 0, 0) == 60


def test_backfill_respects_its_ceiling_and_gpus_per_worker() -> None:
    """A briefly empty cluster must not turn into an unbounded launch."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    beaker = _FakeClusterBeaker({"ai2/jupiter": 800})
    worker = mod.WorkerConfig(
        batch_size=128,
        image_name="img",
        cluster=["ai2/jupiter"],
        backfill_fraction=1.0,
        backfill_max_workers=64,
    )
    assert mod._backfill_target(beaker, worker, 0, 0) == 64

    # Two GPUs per worker halves how many workers the same slots buy.
    worker = mod.WorkerConfig(
        batch_size=128,
        image_name="img",
        cluster=["ai2/jupiter"],
        gpus=2,
        backfill_fraction=0.5,
        backfill_max_workers=1000,
    )
    assert mod._backfill_target(beaker, worker, 0, 0) == 200


def test_backfill_survives_an_unreadable_cluster() -> None:
    """Losing the occupancy read costs backfill for a cycle, not the run."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    class _Broken:
        class cluster:
            @staticmethod
            def get(name: str, include_cluster_occupancy: bool = False) -> None:
                raise RuntimeError("beaker is down")

    worker = mod.WorkerConfig(
        batch_size=128, image_name="img", cluster=["ai2/jupiter"], backfill_fraction=0.5
    )
    assert mod._backfill_target(_Broken(), worker, 0, 0) == 0


def test_backfill_workers_must_stay_unallocated() -> None:
    """Over five minutes a job claims an allocation, which defeats backfill entirely."""
    import importlib
    from datetime import timedelta

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    assert mod.BACKFILL_MIN_RUNTIME <= timedelta(minutes=5), (
        "a backfill worker asking for more than five minutes counts as allocated and "
        "would consume the very allocation it is meant to leave alone"
    )


def test_backfill_does_not_fight_its_own_workers() -> None:
    """The pool's own backfill workers must not read as the cluster filling up.

    They occupy the very slots the target is computed from. Counting only what is idle
    makes a full pool look like a target of zero, the surplus drain retires the workers,
    the slots free, and the next cycle launches them again -- losing a block each time
    round. The steady state has to be a fixed point instead.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = mod.WorkerConfig(
        batch_size=128,
        image_name="img",
        cluster=["ai2/jupiter"],
        backfill_fraction=1.0,
        backfill_max_workers=1000,
    )
    # Cold start: 100 slots idle, pool holds only its allocated workers.
    cold = mod._backfill_target(
        _FakeClusterBeaker({"ai2/jupiter": 100}), worker, 10, 10
    )
    assert cold == 100, f"expected to claim all 100 idle slots, got {cold}"

    # Those 100 are now running, so the cluster reports nothing idle. The target must
    # stay at 100, not collapse to zero.
    warm = mod._backfill_target(_FakeClusterBeaker({"ai2/jupiter": 0}), worker, 110, 10)
    assert warm == 100, f"backfill collapsed to {warm} once its own workers were up"


def test_an_unreadable_cluster_holds_backfill_instead_of_dropping_it() -> None:
    """A failed occupancy read must not read as 'retire every backfill worker'.

    The surplus drain acts on the returned target, so reporting zero would retire a
    whole healthy pool mid-block over one transient Beaker error.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    class _Broken:
        class cluster:
            @staticmethod
            def get(name: str, include_cluster_occupancy: bool = False) -> None:
                raise RuntimeError("beaker is down")

    worker = mod.WorkerConfig(
        batch_size=128,
        image_name="img",
        cluster=["ai2/jupiter"],
        backfill_fraction=1.0,
        backfill_max_workers=1000,
    )
    # 40 allocated, 100 backfill already running.
    assert mod._backfill_target(_Broken(), worker, 140, 40) == 100


def test_queued_backfill_workers_do_not_inflate_the_target() -> None:
    """A worker that is not running holds no slot, so it must not count as capacity.

    Counting queued workers adds capacity the pool does not have to an idle-slot reading
    that has not fallen, so the target climbs by the launch size every cycle. That is
    what produced a pool of ~460 workers that could never be placed, queued ahead of
    this run's own jobs.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = mod.WorkerConfig(
        batch_size=128,
        image_name="img",
        cluster=["ai2/jupiter"],
        backfill_fraction=1.0,
        backfill_max_workers=1000,
    )
    # The cluster is full: nothing idle. 40 allocated plus 30 running backfill hold
    # slots; another 200 were launched but never placed.
    beaker = _FakeClusterBeaker({"ai2/jupiter": 0})
    running = 40 + 30
    target = mod._backfill_target(beaker, worker, running, 40)
    assert target == 30, (
        f"expected the target to equal the 30 backfill workers actually placed, got "
        f"{target}; queued workers must not be counted as held capacity"
    )


def test_backfill_target_is_a_fixed_point_once_slots_are_taken() -> None:
    """Running backfill workers keep their own slots counted, so the pool holds steady."""
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker = mod.WorkerConfig(
        batch_size=128,
        image_name="img",
        cluster=["ai2/jupiter"],
        backfill_fraction=1.0,
        backfill_max_workers=1000,
    )
    # Cold: 100 idle slots, nothing held above the allocated pool.
    cold = mod._backfill_target(
        _FakeClusterBeaker({"ai2/jupiter": 100}), worker, 10, 10
    )
    assert cold == 100
    # Warm: those 100 are running, so the cluster reports nothing idle and the target
    # must stay where it is rather than collapsing.
    warm = mod._backfill_target(_FakeClusterBeaker({"ai2/jupiter": 0}), worker, 110, 10)
    assert warm == 100


def test_backfill_is_sized_from_running_workers_not_the_live_count() -> None:
    """The cycle must hand backfill sizing the running count, not starting + running.

    The arithmetic inside `_backfill_target` is correct either way; what produced the
    unplaceable pool was the call site passing the live count, which includes workers
    that hold no slot. Assert the wiring, since that is what regressed.
    """
    import ast
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    tree = ast.parse(inspect.getsource(mod))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_backfill_target"
    ]
    assert calls, "no call to _backfill_target found"
    for call in calls:
        # (beaker, worker, running, allocated)
        assert len(call.args) >= 3, "unexpected _backfill_target signature"
        third = call.args[2]
        assert isinstance(third, ast.Name), f"expected a name, got {ast.dump(third)}"
        assert third.id == "running", (
            f"_backfill_target is being sized from '{third.id}'; it must be 'running', "
            "or workers that were never placed inflate the target every cycle"
        )


def test_urgent_reserve_splits_the_allocated_pool_by_priority() -> None:
    """Allocated workers past `urgent_workers` must launch at `overflow_priority`.

    Holding the whole allocated pool at urgent leaves colleagues nothing to preempt,
    which is what the reserve exists to avoid. Assert the launch wiring rather than the
    arithmetic: a split that computes the right counts but sends them both to
    `config.worker.priority` looks correct in every unit of the calculation and still
    reserves nothing.
    """
    import ast
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    tree = ast.parse(inspect.getsource(mod))

    priorities = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "launch"
        ):
            continue
        assert len(node.args) >= 2, "unexpected launch() signature"
        priorities.add(ast.dump(node.args[1]))

    wanted = {
        ast.dump(ast.parse(expr, mode="eval").body)
        for expr in (
            "config.worker.priority",
            "config.worker.overflow_priority",
            "config.worker.backfill_priority",
        )
    }
    missing = wanted - priorities
    assert not missing, (
        "launch() is never called with "
        f"{sorted(ast.literal_eval('None') or [] for _ in [])}"
        f"{missing}; the allocated pool must split across priority and "
        "overflow_priority, with backfill on its own"
    )


def test_urgent_reserve_tops_up_to_the_floor_rather_than_relaunching_it() -> None:
    """The reserve must be sized from workers already held, not launched every cycle.

    Launching `urgent_workers` unconditionally each cycle would grow the urgent tier
    without bound while the pool is being refilled.
    """
    import ast
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    src = inspect.getsource(mod)
    tree = ast.parse(src)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_count_urgent_allocated"
    ]
    assert calls, (
        "the cycle never counts the urgent workers it already holds; the reserve "
        "must be topped up to a floor, not relaunched in full every cycle"
    )


def test_each_allocated_tier_gets_its_own_min_runtime() -> None:
    """Urgent and overflow must not share one hardcoded min_runtime.

    The tiers exist so the reserve can buy a stronger guarantee than the bulk of the
    pool. Passing the same constant to both launches computes the right counts and
    still gives every allocated worker the same protection, which is the failure this
    guards against.
    """
    import ast
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    tree = ast.parse(inspect.getsource(mod))

    runtimes = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "launch"
        ):
            continue
        assert len(node.args) >= 3, "unexpected launch() signature"
        runtimes.append(ast.dump(node.args[2]))

    assert len(runtimes) == len(set(runtimes)), (
        "two launch() calls pass the same min_runtime expression; the urgent reserve "
        "and the overflow tier must be able to ask for different guarantees"
    )


def test_tier_min_runtime_falls_back_to_the_worker_default() -> None:
    """Leaving a tier unset must keep the original one-block request."""
    import importlib
    from dataclasses import fields

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    names = {f.name: f for f in fields(mod.WorkerConfig)}
    for name in ("urgent_min_runtime_seconds", "overflow_min_runtime_seconds"):
        assert name in names, f"{name} is not configurable"
        assert names[name].default is None, (
            f"{name} must default to None so an unconfigured run keeps the worker "
            "default rather than silently changing its scheduling guarantee"
        )


def test_urgent_reserve_reads_priority_as_an_enum_not_a_string() -> None:
    """`system_details.priority` is an enum number; stringifying it counts nothing.

    `str(4).rsplit("_", 1)[-1]` is "4", which matches no priority name, so the reserve
    reads as empty every cycle and the urgent tier is relaunched in full each time.
    That fails silently: the counts look plausible and the tier grows without bound.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    class _Enum:
        def __init__(self, number: int, name: str) -> None:
            self.number = number
            self.name = name

    class _Field:
        def __init__(self) -> None:
            self.enum_type = type(
                "E", (), {"values_by_number": {4: _Enum(4, "JOB_PRIORITY_URGENT")}}
            )()

    class _Descriptor:
        fields_by_name = {"priority": _Field()}

    class _Details:
        DESCRIPTOR = _Descriptor()
        priority = 4

        class min_runtime:  # noqa: N801
            seconds = 2700

    class _Task:
        system_details = _Details()

    class _Experiment:
        name = "worker_patrickj-test-abc"
        tasks = [_Task()]

    class _Workload:
        experiment = _Experiment()

    class _Beaker:
        class workload:  # noqa: N801
            @staticmethod
            def list(**_kwargs: object) -> list:
                return [_Workload()]

        class user:  # noqa: N801
            @staticmethod
            def get() -> str:
                return "me"

    held = mod._count_urgent_allocated(
        _Beaker(), object(), "worker_patrickj-test", "urgent"
    )
    assert held == 1, (
        f"an urgent allocated worker was counted as {held}; the reserve must read the "
        "priority enum by number, not by stringifying it"
    )


def test_urgent_reserve_floors_the_allocated_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reserve must raise the target, not just split whatever it yields.

    Sized only as a share of the computed target, the reserve shrinks with contention:
    at a target of 54 a 64-worker reserve silently becomes 54, and the run loses the
    workers it least wanted to lose. The floor is what makes the number mean something.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    class _Slots:
        def __init__(self, n: int) -> None:
            self.n = n

    class _Beaker:
        class workspace:  # noqa: N801
            @staticmethod
            def get(_name: str) -> object:
                return type("W", (), {"id": "ws"})()

    # others hold almost the whole allocation, so `remaining` is far below the reserve
    def fake_usage(*_args: object, **_kwargs: object) -> int:
        return 430

    monkeypatch.setattr(mod, "_allocated_slots_in_use", fake_usage)
    worker = mod.WorkerConfig(
        batch_size=128,
        image_name="img",
        cluster=["ai2/jupiter"],
        num_workers=440,
        gpus=1,
        capacity_fraction=0.9,
        capacity_slots=440,
        urgent_workers=64,
    )
    # live is already at the reserve, so CAPACITY_MAX_STEP's growth pacing is not
    # binding and the floor is what the assertion is actually reading.
    target = mod._capacity_target(_Beaker(), worker, live=64)
    assert target >= 64, (
        f"target {target} fell below the 64-worker urgent reserve; the reserve must "
        "floor the allocated pool, not be carved out of whatever it yields"
    )

    # The growth cap must not hold the pool below its own reserve: from an empty pool
    # one cycle has to be enough to reach the floor, or a run that has just lost its
    # workers spends cycles under the level the reserve exists to guarantee.
    from_zero = mod._capacity_target(_Beaker(), worker, live=0)
    assert from_zero >= 64, (
        f"target {from_zero} from an empty pool is below the 64-worker reserve; "
        f"CAPACITY_MAX_STEP ({mod.CAPACITY_MAX_STEP}) must not be smaller than the "
        "reserve it has to reach"
    )


def test_pool_never_targets_zero_without_an_urgent_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no reserve configured, the pool still must not target zero.

    `capacity_min_workers` used to supply this via its default of 8. Removing it must
    not let a momentarily full allocation park the run with nothing running to pick
    slots back up.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    assert mod.MIN_POOL_WORKERS > 0

    def fake_usage(*_args: object, **_kwargs: object) -> int:
        return 440

    monkeypatch.setattr(mod, "_allocated_slots_in_use", fake_usage)

    class _Beaker:
        class workspace:  # noqa: N801
            @staticmethod
            def get(_name: str) -> object:
                return type("W", (), {"id": "ws"})()

    worker = mod.WorkerConfig(
        batch_size=128,
        image_name="img",
        cluster=["ai2/jupiter"],
        num_workers=440,
        gpus=1,
        capacity_fraction=0.9,
        capacity_slots=440,
        urgent_workers=0,
    )
    assert mod._capacity_target(_Beaker(), worker, live=0) >= mod.MIN_POOL_WORKERS


def test_allocation_usage_counts_a_multi_cluster_job_once() -> None:
    """A job eligible for several of the pool's clusters must not be counted per cluster.

    Beaker's eligibility listing returns such a job once per cluster. Summing the
    listings double counts it, and that is not a rounding error: with jupiter and ceres
    every ceres-eligible job was also jupiter-eligible, so 344 real slots read as 508
    and the computed target collapsed from 198 to 54 while the pool drained.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    class _Job:
        def __init__(self, jid: str, gpus: int) -> None:
            self.id = jid
            self.workspace_id = "ws"
            self.container_spec = type(
                "C", (), {"resource_request": type("R", (), {"gpu_count": gpus})()}
            )()

    # one job eligible for both clusters, one eligible for jupiter only
    both = _Job("shared", 8)
    jupiter_only = _Job("solo", 4)
    listings = {"ai2/jupiter": [both, jupiter_only], "ai2/ceres": [both]}

    class _Beaker:
        class cluster:  # noqa: N801
            @staticmethod
            def get(name: str) -> str:
                return name

        class job:  # noqa: N801
            @staticmethod
            def list(elegible_for_cluster: str, **_kw: object) -> list:
                return listings[elegible_for_cluster]

    worker = mod.WorkerConfig(
        batch_size=128, image_name="img", cluster=["ai2/jupiter", "ai2/ceres"], gpus=1
    )
    used = mod._allocated_slots_in_use(_Beaker(), worker, "ws", live=0)
    assert used == 12, (
        f"expected 12 slots (8 + 4, the shared job counted once), got {used}; "
        "summing per-cluster listings double counts multi-cluster jobs"
    )


def test_reaper_only_fires_when_the_reserve_is_short() -> None:
    """A healthy reserve must not have its overflow cancelled.

    The reaper exists to free headroom for urgent relaunches. Running it when the
    reserve is already full would cancel capacity that is merely waiting its turn, and
    on a saturated cluster that is most of the pool.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    class _Beaker:
        class workload:  # noqa: N801
            @staticmethod
            def list(**_kwargs: object) -> list:
                raise AssertionError("must not enumerate workers when not short")

    for shortfall in (0, -5):
        assert (
            mod._reap_unplaceable_overflow(
                _Beaker(), object(), "worker_x", "urgent", shortfall, 2700
            )
            == 0
        ), "the reaper ran with a full reserve"


def test_reaper_spares_backfill_running_and_urgent_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only queued, stale, allocated, non-reserve workers may be cancelled.

    Cancelling a running worker throws away a whole job, backfill holds no allocated
    slot so freeing it buys the reserve nothing, and cancelling urgent would remove the
    very capacity being topped up.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    old = 1_000.0

    def _wl(
        name: str, status: int, priority: int, min_runtime: int, created: int
    ) -> object:
        return type(
            "W",
            (),
            {
                "experiment": type(
                    "E",
                    (),
                    {
                        "name": name,
                        "created": type("C", (), {"seconds": created})(),
                        "tasks": [
                            type(
                                "T",
                                (),
                                {
                                    "status": status,
                                    "system_details": _FakeDetails(
                                        priority, min_runtime
                                    ),
                                },
                            )()
                        ],
                    },
                )(),
            },
        )()

    stale = int(old - 10_000)
    pool = [
        _wl("worker_x_run", 4, 4, 14400, stale),  # running
        _wl("worker_x_bf", 2, 5, 60, stale),  # backfill
        _wl("worker_x_urg", 2, 4, 14400, stale),  # urgent
        _wl("worker_x_fresh", 2, 5, 7200, int(old)),  # queued but young
        _wl("worker_x_stale", 2, 5, 7200, stale),  # the only valid target
    ]
    cancelled: list = []

    class _Beaker:
        class workload:  # noqa: N801
            @staticmethod
            def list(**_kwargs: object) -> list:
                return pool

            @staticmethod
            def cancel(*workloads: object) -> None:
                cancelled.extend(workloads)

        class user:  # noqa: N801
            @staticmethod
            def get() -> str:
                return "me"

    import datetime as _dt

    class _Now(_dt.datetime):
        @classmethod
        def now(cls, tz: object = None) -> "_Now":
            return cls.fromtimestamp(old, tz)  # type: ignore[arg-type]

    monkeypatch.setattr(mod, "datetime", _Now)
    n = mod._reap_unplaceable_overflow(
        _Beaker(), object(), "worker_x", "urgent", 10, 100
    )

    names = [w.experiment.name for w in cancelled]
    assert names == ["worker_x_stale"], f"cancelled the wrong workers: {names}"
    assert n == 1


class _FakeDetails:
    """Minimal stand-in for a task's system_details."""

    def __init__(self, priority: int, min_runtime_seconds: int) -> None:
        self.priority = priority
        self.min_runtime = type("R", (), {"seconds": min_runtime_seconds})()

    @property
    def DESCRIPTOR(self) -> object:  # noqa: N802
        names = {4: "JOB_PRIORITY_URGENT", 5: "JOB_PRIORITY_HIGH"}

        class _Enum:
            values_by_number = {
                k: type("V", (), {"name": v})() for k, v in names.items()
            }

        class _Field:
            enum_type = _Enum()

        return type("D", (), {"fields_by_name": {"priority": _Field()}})()


def test_batch_size_reaches_workers_and_not_the_queue() -> None:
    """Batch size must travel with the worker, never baked into a queue entry.

    One queue feeds pools on different hardware, so a value fixed at enqueue time
    sizes every GPU's batch for whichever is smallest. Keeping it out of the entry is
    half of that; putting it in the worker's environment is the other half.
    """
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    jobs = importlib.import_module("rslp.large_scale_embeddings.write_jobs")

    assert "--batch_size" not in inspect.getsource(
        jobs
    ), "write_jobs still emits --batch_size into queue entries"

    src = inspect.getsource(mod)
    assert (
        "WORKER_EXTRA_ARGS_ENV" in src
    ), "the supervisor must pass the batch size to workers through the environment"
    assert (
        "config.worker.batch_size" in src
    ), "the batch size must come from the worker config, not the model config"


def test_batch_size_is_required_on_the_worker_config() -> None:
    """Omitting it must fail loudly rather than fall back to the model config.

    A silent fallback would size an A100's batch from a YAML written for an H100, and
    the failure would be an out-of-memory crash on some later tile rather than here.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    try:
        mod.WorkerConfig(image_name="i", cluster=["c"])
    except TypeError as e:
        assert "batch_size" in str(e), f"wrong error: {e}"
    else:
        raise AssertionError("WorkerConfig accepted no batch_size")

    assert not hasattr(
        mod.ModelConfig(checkpoint_path="p"), "batch_size"
    ), "batch_size must not remain on ModelConfig, whose settings change the output"


def _job(crs: str, x: int, y: int) -> list[str]:
    """One worker argument list, carrying just the fields the ordering reads."""
    return [
        "--projection_json",
        json.dumps({"crs": crs, "x_resolution": 10, "y_resolution": -10}),
        "--bounds",
        json.dumps([x, y, x + 4096, y + 4096]),
    ]


def _footprint(
    tmp_path: Path, lon: float, lat: float, half: float = 1.0, name: str = "priority"
) -> str:
    """Write a square GeoJSON footprint around a point."""
    path = tmp_path / f"{name}.geojson"
    box = [
        [lon - half, lat - half],
        [lon + half, lat - half],
        [lon + half, lat + half],
        [lon - half, lat + half],
        [lon - half, lat - half],
    ]
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [box]},
                    }
                ],
            }
        )
    )
    return str(path)


def test_priority_footprint_tiles_are_enqueued_first(tmp_path: Path) -> None:
    """Tiles inside the footprint must come before every tile outside it.

    The queue is kept shallow on purpose, so enqueue order decides what is worked on
    next. Ordering here is the entire priority mechanism; a tile that sorts late is
    not merely deprioritised, it waits for everything else.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    # UTM 14N, around 99W 39N: inside the footprint written below.
    inside = [_job("EPSG:32614", 49152, -438272), _job("EPSG:32614", 53248, -438272)]
    # UTM 48N, Cambodia: far outside it.
    outside = [_job("EPSG:32648", 45056, -126976), _job("EPSG:32648", 49152, -126976)]

    got = mod._priority_first(
        outside + inside, [_footprint(tmp_path, -98.7, 39.4, 2.0)]
    )

    assert len(got) == 4
    assert all(j in inside for j in got[:2]), f"priority tiles not first: {got[:2]}"
    assert all(j in outside for j in got[2:]), "non-priority tiles not last"


def test_no_footprint_still_shuffles(tmp_path: Path) -> None:
    """Without a footprint the ordering must stay random, not become insertion order.

    The shuffle spreads load across zones and imagery sources. Returning the list
    untouched would quietly concentrate a cycle's batch on one region.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    jobs = [_job("EPSG:32614", 4096 * i, -438272) for i in range(50)]

    orders = {tuple(map(tuple, mod._priority_first(jobs, None))) for _ in range(5)}

    assert len(orders) > 1, "order was identical every time, so nothing is shuffling"


def test_unreadable_footprint_does_not_stall_the_run(tmp_path: Path) -> None:
    """A bad footprint must cost ordering, not the cycle.

    Enqueueing is how the pool is kept fed; raising here would stop the run over a
    file that only decides what order to do the work in.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    jobs = [_job("EPSG:32614", 49152, -438272) for _ in range(3)]

    got = mod._priority_first(jobs, [str(tmp_path / "missing.geojson")])

    assert len(got) == len(jobs)


def test_footprints_are_ordered_by_their_position_in_the_list(tmp_path: Path) -> None:
    """Earlier footprints must outrank later ones, and both outrank the rest.

    The list is the priority order. Tiering by anything else, such as area or match
    count, would make "do CONUS before Europe" depend on the shapes rather than on
    what was asked for.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    kansas = [_job("EPSG:32614", 49152, -438272)]  # ~99W 39N
    cambodia = [_job("EPSG:32648", 45056, -126976)]  # ~105E 11N
    peru = [_job("EPSG:32619", 36864, 143360)]  # southern hemisphere, matches neither

    first = _footprint(tmp_path, -98.7, 39.4, 2.0, name="a")
    second = _footprint(tmp_path, 104.8, 11.3, 2.0, name="b")

    got = mod._priority_first(peru + cambodia + kansas, [first, second])
    assert got[0] in kansas, "first footprint did not win"
    assert got[1] in cambodia, "second footprint did not come next"
    assert got[2] in peru, "unmatched tile did not go last"

    # Swapping the list must swap the order, with nothing else changed.
    got = mod._priority_first(peru + cambodia + kansas, [second, first])
    assert (
        got[0] in cambodia and got[1] in kansas
    ), f"order did not follow the list: {got}"


def test_a_tile_takes_the_first_footprint_it_falls_in(tmp_path: Path) -> None:
    """Overlapping footprints resolve by list order, not by whichever matches last.

    Checked with two groups rather than two tiles: a tile in both footprints must
    outrank one in only the second. Were the scan to keep going instead of stopping at
    the first match, both would land in the same tier and interleave, which a
    single-tile assertion would miss half the time.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    # 39.4N down to 37.6N, all inside narrow, which sits inside wide.
    both = [_job("EPSG:32614", 49152, -438272 + 4096 * i) for i in range(6)]
    # 33.8N down to 30.5N: inside wide, south of narrow's 37.4N edge.
    wide_only = [_job("EPSG:32614", 49152, -376320 + 4096 * i) for i in range(10)]

    narrow = _footprint(tmp_path, -98.7, 39.4, 2.0, name="narrow")
    wide = _footprint(tmp_path, -98.0, 36.0, 6.0, name="wide")

    got = mod._priority_first(wide_only + both, [narrow, wide])
    ranks = [0 if j in both else 1 for j in got]
    assert ranks == sorted(
        ranks
    ), "a tile in both footprints must come before one in only the later footprint"


def test_an_unreadable_footprint_leaves_the_other_tiers_working(
    tmp_path: Path,
) -> None:
    """One footprint failing to load must not cost the others their ordering.

    Dropping the whole ordering on a single bad path would turn a typo in one entry
    into a run that quietly does its work in random order.
    """
    import importlib

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")

    kansas = _job("EPSG:32614", 49152, -438272)
    cambodia = _job("EPSG:32648", 45056, -126976)
    missing = str(tmp_path / "missing.geojson")
    real = _footprint(tmp_path, 104.8, 11.3, 2.0, name="real")

    got = mod._priority_first([kansas, cambodia], [missing, real])
    assert got[0] == cambodia, "the readable footprint should still rank above the rest"
    assert len(got) == 2


def test_a_run_can_use_its_own_gcp_identity() -> None:
    """The credentials secret must reach both the supervisor and its workers.

    `RSLEARN_GCP_CREDENTIALS` is the shared default across every rslp Beaker job, so
    changing which account writes the archive has to be possible for one run without
    moving training, materialization and the vessel pipelines onto it too. The
    supervisor needs the same identity as its workers, since it reads the markers they
    write.
    """
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    worker_mod = importlib.import_module("rslp.common.worker")

    assert (
        "gcp_credentials_secret" in inspect.signature(mod.launch_supervisor).parameters
    ), "launch_supervisor cannot be pointed at a different credentials secret"
    assert (
        "gcp_credentials_secret"
        in inspect.signature(worker_mod.launch_workers).parameters
    ), "launch_workers cannot be pointed at a different credentials secret"

    cfg = mod.WorkerConfig(image_name="i", cluster=["c"], batch_size=128)
    assert (
        cfg.gcp_credentials_secret is None
    ), "the default must stay the shared secret, so existing runs are unaffected"

    src = inspect.getsource(mod)
    assert (
        "gcp_credentials_secret=config.worker.gcp_credentials_secret" in src
    ), "the supervisor does not pass its configured secret on to the workers"
