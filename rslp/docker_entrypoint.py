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
    import rslearn.main

    rslearn.main.main()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
