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
    assert not mine.startswith(theirs) and not theirs.startswith(mine), (
        "one queue's prefix matches another's, so their worker counts would collide"
    )
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

    for env in (None, {}, {"SOMETHING_ELSE": "1"}):
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
