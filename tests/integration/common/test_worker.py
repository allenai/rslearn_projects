"""Test common.worker functionality."""

import os
import pathlib
import time

from rslp.common.worker import worker_pipeline, write_jobs


def test_idle_timeout() -> None:
    """Verify that worker exits within about the idle timeout."""
    idle_timeout = 2
    start_time = time.time()
    worker_pipeline(
        project_id=os.environ["TEST_PUBSUB_PROJECT"],
        subscription_id=os.environ["TEST_PUBSUB_SUBSCRIPTION"],
        idle_timeout=idle_timeout,
    )
    end_time = time.time()
    elapsed = end_time - start_time
    # Use 3 here in case worker decides to sleep twice (and then some extra time
    # elapses).
    assert elapsed >= idle_timeout and elapsed <= 3 * idle_timeout


def test_single_task(tmp_path: pathlib.Path) -> None:
    """Check that worker can do one task."""
    # Use a task that writes to this filename.
    dst_fname = tmp_path / "test_file"
    # Write the task to the test topic.
    job_args = [
        str(dst_fname),
        # The contents to write.
        "abc",
    ]
    write_jobs(
        project_id=os.environ["TEST_PUBSUB_PROJECT"],
        topic_id=os.environ["TEST_PUBSUB_TOPIC"],
        rslp_project="common",
        rslp_workflow="write_file",
        args_list=[job_args],
    )
    # Run the worker.
    worker_pipeline(
        project_id=os.environ["TEST_PUBSUB_PROJECT"],
        subscription_id=os.environ["TEST_PUBSUB_SUBSCRIPTION"],
        idle_timeout=2,
    )
    # Verify that the file was created.
    assert dst_fname.exists()
