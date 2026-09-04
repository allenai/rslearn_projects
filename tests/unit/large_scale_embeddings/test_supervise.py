def test_every_kwarg_the_cycle_reads_is_actually_passed() -> None:
    """The child process's kwargs dict must carry everything the cycle reads from it.

    supervise runs each cycle in a spawned process and hands it one explicitly built
    dict, so a parameter added to supervise's signature reaches the cycle only if it is
    also added to that literal. Miss it and the option is accepted, documented, and
    silently ignored: `worker_idle_seconds` shipped that way, and the run looked correct
    while every worker still used the ten-second default. Comparing what is read against
    what is packed catches the whole class rather than one instance of it.
    """
    import ast
    import importlib
    import inspect

    # importlib, not `from ... import supervise`: the package re-exports the function
    # of that name, so the plain import binds the function and getsource then returns
    # one function body. The dict lives in it and the reads do not, which makes the
    # comparison below vacuously true -- this test passed against the very defect it
    # was written for until the import was fixed.
    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    assert inspect.ismodule(mod), "expected the module, got the re-exported function"

    tree = ast.parse(inspect.getsource(mod))

    packed: set[str] = set()
    for node in ast.walk(tree):
        # Annotated (`kwargs: dict[str, Any] = {...}`) as well as plain: the dict is
        # written with an annotation today, and a test that silently matches neither
        # form would pass on an empty set.
        if isinstance(node, ast.AnnAssign):
            targets = [node.target.id] if isinstance(node.target, ast.Name) else []
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        else:
            continue
        if "kwargs" not in targets or not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                packed.add(key.value)
    assert packed, "could not find the kwargs dict literal"

    read: set[str] = set()
    for node in ast.walk(tree):
        # kwargs["name"]
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "kwargs"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            read.add(node.slice.value)
        # kwargs.get("name")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "kwargs"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            read.add(node.args[0].value)

    # A silently empty read set is how this test fooled itself once already.
    assert len(read) > 10, f"only found {len(read)} kwargs reads; the walk is wrong"
    missing = sorted(read - packed)
    assert not missing, f"read from kwargs but never packed into it: {missing}"


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
    refills is that work plus `cycle_seconds`, bounded above by `cycle_budget_seconds`
    because the parent kills a cycle that overruns. A worker that idles out sooner has
    to be relaunched and pay container start again.
    """
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    defaults = {
        name: param.default
        for name, param in inspect.signature(mod.supervise).parameters.items()
    }
    idle = defaults["worker_idle_seconds"]
    assert idle is not None, (
        "worker_idle_seconds defaults to None, which hands the worker its own "
        "ten-second timeout, so it quits the moment the queue drains"
    )
    worst_case = defaults["cycle_budget_seconds"] + defaults["cycle_seconds"]
    assert idle >= worst_case, (
        f"a worker idles out after {idle}s but a cycle can take up to {worst_case}s "
        "to come back round, so the pool empties between refills"
    )


def test_workers_always_get_the_gdal_billing_project() -> None:
    """`supervise` must merge the GDAL worker env, not rely on its caller to.

    The requester-pays USGS Landsat mirror needs GDAL handed a billing project via
    GS_USER_PROJECT; without it every Landsat read fails with an HTTP 400 that names no
    variable, and the run carries on producing embeddings with the Landsat inputs
    missing. Silent degradation, not a crash.

    This lived in `run_all` only, so a supervisor launched directly through
    `launch_supervisor` skipped it entirely. Five diagnostic runs went out that way.
    """
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    assert "GS_USER_PROJECT" in mod.DEFAULT_WORKER_ENV_VARS
    src = inspect.getsource(mod.supervise)
    assert "DEFAULT_WORKER_ENV_VARS," in src, (
        "supervise packs worker_env_vars without merging the defaults, so a caller "
        "that does not pass them launches workers with no billing project"
    )


def test_an_explicit_worker_env_var_still_wins() -> None:
    """Merging defaults must not stop a caller overriding one.

    The merge order matters: defaults first, caller second. Reversed, a run could not
    point at a different billing project.
    """
    import importlib
    import inspect

    mod = importlib.import_module("rslp.large_scale_embeddings.supervise")
    src = inspect.getsource(mod.supervise)
    i = src.index("DEFAULT_WORKER_ENV_VARS,")
    j = src.index("(worker_env_vars or {}),")
    assert i < j, "caller-supplied worker_env_vars must be spread after the defaults"
