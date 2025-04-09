"""Test common.worker functionality."""

import os
import pathlib
import time
from collections.abc import Callable

from rslp.common.worker import worker_pipeline, write_jobs


class TestWorker:
    """Test the rslp.common.worker module."""

    def setup_method(self, method: Callable) -> None:
        """Flush messages that may be in the test subscription."""
        print("begin flushing messages for test subscription")
        worker_pipeline(
            project_id=os.environ["TEST_PUBSUB_PROJECT"],
            subscription_id=os.environ["TEST_PUBSUB_SUBSCRIPTION"],
            idle_timeout=1,
            flush_messages=True,
        )
        print("done flushing messages")

    def test_idle_timeout(self) -> None:
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

    def test_single_task(self, tmp_path: pathlib.Path) -> None:
        """Check that worker can do one task."""
        # Use a task that writes to this filename.
        dst_fname = tmp_path / "test_file"
        # Write the task to the test topic.
        job_args = [
            "--fname",
            str(dst_fname),
            "--contents",
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
            idle_timeout=1,
        )
        # Verify that the file was created.
        assert dst_fname.exists()
