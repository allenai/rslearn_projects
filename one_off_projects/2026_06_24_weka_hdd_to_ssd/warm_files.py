"""Multiprocess file warmer to pull a tree from Weka HDD into SSD cache.

Reading a file's bytes promotes it from the Weka HDD tier into the SSD cache, so
this script simply reads every file under a root path. Each directory is scanned
by a single ``os.scandir``; its direct files are read in full (bytes discarded)
and its subdirectories are handed back to be scanned as their own work items by
other workers.

Structure mirrors ``../2026_06_10_disk_usage/disk_usage.py`` but stripped down:
- The main process owns a DFS *stack* of pending directories. Newly discovered
  subdirs are pushed and popped next, so the in-memory frontier stays ~depth x
  branching instead of an entire BFS level.
- Directories are dispatched/returned in *batches* (default 100), cutting the
  number of pickled IPC messages versus one message per directory.
- An in-flight cap (``--max_inflight`` batches) provides backpressure so the
  queues never balloon. Because we self-throttle, the queues stay bounded and
  no ``put`` ever blocks, which avoids the classic two-queue deadlock.

There is no fanout collapse, checkpointing, or resumability: this is a one-shot
warm-up that only reports aggregate stats to stderr.

Example:
    python warm_files.py --root /weka/dfive-default/some/dataset --workers 128
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import random
import sys
import time

# Result for one scanned directory: how much was read, any errors, and the
# subdirectories discovered (to be scanned next).
DirResult = tuple[str, int, int, list[str], list[str]]


def scan_and_read(path: str, chunk_size: int) -> DirResult:
    """Read every direct file in ``path``; return stats and its subdirectories.

    Files are read fully in ``chunk_size`` chunks and the bytes discarded; the
    point is the read itself (which pulls the file from HDD into SSD). Never
    raises; OSErrors are collected into the returned ``errors`` list.
    """
    errors: list[str] = []
    files_read = 0
    bytes_read = 0
    subdirs: list[str] = []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        subdirs.append(e.path)
                        continue
                    if not e.is_file(follow_symlinks=False):
                        continue
                    with open(e.path, "rb", buffering=0) as f:
                        while True:
                            buf = f.read(chunk_size)
                            if not buf:
                                break
                            bytes_read += len(buf)
                    files_read += 1
                except OSError as err:
                    errors.append(f"{e.path}: {err}")
    except OSError as err:
        errors.append(f"{path}: {err}")

    # Shuffle so that multiple concurrent runs of this warmer (e.g. the same job
    # launched on several hosts) traverse the tree in different orders. Each run
    # still does a DFS and eventually reads everything, but they avoid all
    # hammering the same files at the same time.
    random.shuffle(subdirs)

    return path, files_read, bytes_read, errors, subdirs


def worker(task_queue: mp.Queue, result_queue: mp.Queue, chunk_size: int) -> None:
    """Pull batches of directories, read each, return per-directory results.

    Each task is ``(batch_id, items)``. The result is ``(batch_id, [DirResult,
    ...])``; the ``batch_id`` lets main drop the batch from its in-flight set.
    """
    while True:
        task = task_queue.get()
        if task is None:  # sentinel -> shut down
            break
        batch_id, items = task
        out = [scan_and_read(path, chunk_size) for path in items]
        result_queue.put((batch_id, out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory to warm.")
    parser.add_argument(
        "--workers", type=int, default=128, help="Number of worker processes."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Directories dispatched per task message (amortizes IPC overhead).",
    )
    parser.add_argument(
        "--max_inflight",
        type=int,
        default=0,
        help="Max batches dispatched but not yet returned (backpressure). "
        "0 means 2 x workers.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4 * 1024 * 1024,
        help="Read size in bytes per chunk when reading each file.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    batch_size = max(1, args.batch_size)
    chunk_size = max(1, args.chunk_size)
    max_inflight = args.max_inflight if args.max_inflight > 0 else 2 * args.workers

    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=worker, args=(task_queue, result_queue, chunk_size))
        for _ in range(args.workers)
    ]
    for p in procs:
        p.start()

    # Single DFS frontier of directories pending a scan.
    stack: list[str] = [root]

    # Batches dispatched but not yet returned, kept only to know when we're done.
    inflight: dict[int, list[str]] = {}
    next_id = 0

    dirs_done = 0
    files_read = 0
    bytes_read = 0
    error_count = 0

    def dispatch() -> None:
        nonlocal next_id
        while len(inflight) < max_inflight and stack:
            n = min(batch_size, len(stack))
            batch = [stack.pop() for _ in range(n)]  # pop -> popped next == DFS
            inflight[next_id] = batch
            task_queue.put((next_id, batch))
            next_id += 1

    last_log = time.monotonic()
    try:
        dispatch()
        while inflight:
            try:
                batch_id, results = result_queue.get(timeout=60.0)
            except queue.Empty:
                continue

            inflight.pop(batch_id, None)
            for _path, n_files, n_bytes, errors, subdirs in results:
                dirs_done += 1
                files_read += n_files
                bytes_read += n_bytes
                error_count += len(errors)
                for err in errors:
                    print(f"error: {err}", file=sys.stderr)
                stack.extend(subdirs)
            dispatch()

            now = time.monotonic()
            if now - last_log >= 5.0:
                print(
                    f"dirs={dirs_done} files={files_read} "
                    f"GiB={bytes_read / 1024**3:.2f} "
                    f"stack={len(stack)} inflight={len(inflight)}",
                    file=sys.stderr,
                )
                last_log = now
    finally:
        for _ in procs:
            task_queue.put(None)  # sentinels
        for p in procs:
            p.join()

    print(
        f"done: read {files_read} files "
        f"({bytes_read / 1024**3:.2f} GiB) across {dirs_done} dirs, "
        f"{error_count} errors",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
