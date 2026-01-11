"""Docker entrypoint for rslp."""

import multiprocessing


def main() -> None:
    """Docker entrypoint for rslp.

    Downloads the code from GCS before running the job.

    The RSLP_PROJECT and RSLP_EXPERIMENT environmental variables must be set.
    """
    import os

    project_id = os.environ["RSLP_PROJECT"]
    experiment_id = os.environ["RSLP_EXPERIMENT"]
    from rslp.launcher_lib import download_code

    download_code(project_id, experiment_id)
    import rslp.rslearn_main

    rslp.rslearn_main.main()


if __name__ == "__main__":
    # Use spawn instead of forkserver to avoid "too many fds" error
    # when dataset has many open file handles
    multiprocessing.set_start_method("spawn")
    main()
