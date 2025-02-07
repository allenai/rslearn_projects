"""An example workflow that just writes a file.

This is used for testing. It needs to be under rslp/ so that it can be used as a
workflow.
"""


def write_file(fname: str, contents: str) -> None:
    """Write the contents to the file.

    Args:
        fname: the filename to write.
        contents: the data to write to the file.
    """
    with open(fname, "w") as f:
        f.write(contents)
