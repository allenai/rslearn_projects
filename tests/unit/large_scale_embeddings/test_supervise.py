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
