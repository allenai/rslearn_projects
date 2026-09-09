"""Multiprocess disk usage scanner.

Each directory is scanned by a single ``os.scandir`` (summing the bytes of files
directly in it); its subdirectories are handed back to be scanned as their own
work items. One JSONL line is emitted per directory (direct files only), written
from a single process, so summing all lines gives the grand total.

Design notes (why explicit workers instead of multiprocessing.Pool):
- The main process owns a DFS *stack* of pending directories. Newly discovered
  subdirs are pushed and popped next, so the in-memory frontier stays ~depth x
  branching instead of an entire BFS level (which can be millions of dirs).
- Directories are dispatched/returned in *batches* (default 100), cutting the
  number of pickled IPC messages ~100x versus one message per directory.
- An in-flight cap (``--max_inflight`` batches) provides backpressure so the
  queues never balloon. Because we self-throttle, the queues stay bounded and
  no ``put`` ever blocks, which avoids the classic two-queue deadlock.
- Workers do the ``json.dumps`` so serialization happens in parallel and the
  result messages are cheap strings.

Dynamic fan-out collapse: when a folder has more than ``--fanout_threshold``
subfolders, each of those subfolders is scanned as a single full recursive
total (one ``recursive: True`` line, no descendant lines) instead of being
recursed into. This bounds the output file size for high-fan-out trees while
keeping parallelism across the many subfolders.

Resumability: every ``--checkpoint_interval`` seconds the pending frontier (both
stacks plus the items currently in flight) is written atomically to a checkpoint
file. If that file exists at startup we resume from it and append to the JSONL.
Because resuming re-scans the frontier, the same directory may be written more
than once; collapse.py dedups by path, so duplicates are harmless.

Use collapse.py to fold this full output into a bounded tree for the web app.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time

# (path, depth) work items.
WorkItem = tuple[str, int]


def scan_dir(
    path: str, depth: int, fanout_threshold: int
) -> tuple[dict, list[WorkItem], list[WorkItem]]:
    """Scan a single directory's direct contents.

    Returns the (direct-only) entry dict for ``path`` and its subdirectories
    split into normal-scan children and full-scan children: if the folder has
    more than ``fanout_threshold`` subfolders, all of them go to full-scan (each
    becomes one recursive total) instead of being recursed into. Never raises;
    OSErrors are collected into the entry's ``errors`` list.
    """
    errors: list[str] = []
    direct_bytes = 0
    direct_file_count = 0
    subdirs: list[str] = []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        subdirs.append(e.path)
                    else:
                        st = e.stat(follow_symlinks=False)
                        direct_bytes += st.st_size
                        direct_file_count += 1
                except OSError as err:
                    errors.append(f"{e.path}: {err}")
    except OSError as err:
        errors.append(f"{path}: {err}")

    entry = {
        "path": path,
        "depth": depth,
        "recursive": False,
        "direct_file_count": direct_file_count,
        "direct_bytes": direct_bytes,
        "num_subdirs": len(subdirs),
        "errors": errors,
    }
    children = [(sd, depth + 1) for sd in subdirs]
    if len(subdirs) > fanout_threshold:
        return entry, [], children
    return entry, children, []


def full_scan(path: str, depth: int) -> dict:
    """Account an entire subtree in one pass; return one recursive entry."""
    errors: list[str] = []
    total_bytes = 0
    total_file_count = 0

    def on_walk_error(err: OSError) -> None:
        errors.append(f"{err.filename}: {err}")

    for root, _dirs, files in os.walk(path, followlinks=False, onerror=on_walk_error):
        for name in files:
            fpath = os.path.join(root, name)
            try:
                st = os.lstat(fpath)
            except OSError as err:
                errors.append(f"{fpath}: {err}")
                continue
            total_bytes += st.st_size
            total_file_count += 1

    return {
        "path": path,
        "depth": depth,
        "recursive": True,
        "total_file_count": total_file_count,
        "total_bytes": total_bytes,
        "errors": errors,
    }


def worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    fanout_threshold: int,
) -> None:
    """Pull mode-tagged batches, scan each item, return per-item results.

    Each task is ``(batch_id, mode, items)``. The result is
    ``(batch_id, [(json_line, normal_children, full_children), ...])``; the
    ``batch_id`` lets main drop the batch from its in-flight set. Full-scan
    items have no children.
    """
    while True:
        task = task_queue.get()
        if task is None:  # sentinel -> shut down
            break
        batch_id, mode, items = task
        out: list[tuple[str, list[WorkItem], list[WorkItem]]] = []
        for path, depth in items:
            if mode == "full":
                entry = full_scan(path, depth)
                out.append((json.dumps(entry), [], []))
            else:
                entry, normal_children, full_children = scan_dir(
                    path, depth, fanout_threshold
                )
                out.append((json.dumps(entry), normal_children, full_children))
        result_queue.put((batch_id, out))


def load_checkpoint(path: str) -> tuple[list[WorkItem], list[WorkItem]]:
    """Read a checkpoint into (normal_stack, full_stack)."""
    normal: list[WorkItem] = []
    full: list[WorkItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mode, depth, dir_path = json.loads(line)
            if mode == "full":
                full.append((dir_path, depth))
            else:
                normal.append((dir_path, depth))
    return normal, full


def write_checkpoint(
    path: str,
    out_file,
    normal_stack: list[WorkItem],
    full_stack: list[WorkItem],
    inflight_items: dict[int, tuple[str, list[WorkItem]]],
) -> None:
    """Atomically dump the pending frontier (stacks + in-flight) to ``path``.

    The JSONL is flushed first so that, after a crash, everything not in the
    checkpoint is durably present in the output (and anything written after this
    point is regenerated by re-scanning the frontier).
    """
    out_file.flush()
    os.fsync(out_file.fileno())

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for mode, stack in (("scan", normal_stack), ("full", full_stack)):
            for dir_path, depth in stack:
                f.write(json.dumps([mode, depth, dir_path]) + "\n")
        for mode, items in inflight_items.values():
            for dir_path, depth in items:
                f.write(json.dumps([mode, depth, dir_path]) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory to scan.")
    parser.add_argument(
        "--workers", type=int, default=64, help="Number of worker processes."
    )
    parser.add_argument(
        "--output", default="disk_usage.jsonl", help="Output JSONL path."
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
        "--fanout_threshold",
        type=int,
        default=1000,
        help="If a folder has more than this many subfolders, each subfolder is "
        "scanned as a single full recursive total instead of being recursed.",
    )
    parser.add_argument(
        "--full_batch_size",
        type=int,
        default=1,
        help="Full-scan items dispatched per task message. 1 lets each heavy "
        "subtree walk land on its own worker.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path for resuming. Empty means <output>.ckpt.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=float,
        default=1800.0,
        help="Seconds between checkpoint dumps of the pending frontier.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    batch_size = max(1, args.batch_size)
    full_batch_size = max(1, args.full_batch_size)
    max_inflight = args.max_inflight if args.max_inflight > 0 else 2 * args.workers
    checkpoint_path = args.checkpoint or (args.output + ".ckpt")

    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(
            target=worker, args=(task_queue, result_queue, args.fanout_threshold)
        )
        for _ in range(args.workers)
    ]
    for p in procs:
        p.start()

    # Two DFS frontiers owned by main: normal scans and (expensive) full scans.
    if os.path.exists(checkpoint_path):
        normal_stack, full_stack = load_checkpoint(checkpoint_path)
        out_file = open(args.output, "a")
        print(
            f"resuming from {checkpoint_path}: "
            f"{len(normal_stack)} normal + {len(full_stack)} full pending",
            file=sys.stderr,
        )
    else:
        normal_stack = [(root, 0)]
        full_stack = []
        out_file = open(args.output, "w")

    # Batches dispatched but not yet returned, kept so they can be checkpointed
    # (and re-scanned on resume) rather than lost on a crash.
    inflight_items: dict[int, tuple[str, list[WorkItem]]] = {}
    next_id = 0
    written = 0

    def dispatch() -> None:
        nonlocal next_id
        while len(inflight_items) < max_inflight and (normal_stack or full_stack):
            # Prefer full scans so the big subtree walks start early.
            if full_stack:
                n = min(full_batch_size, len(full_stack))
                batch = [full_stack.pop() for _ in range(n)]
                mode = "full"
            else:
                n = min(batch_size, len(normal_stack))
                batch = [normal_stack.pop() for _ in range(n)]
                mode = "scan"
            inflight_items[next_id] = (mode, batch)
            task_queue.put((next_id, mode, batch))
            next_id += 1

    last_dump = time.monotonic()
    try:
        dispatch()
        while inflight_items:
            now = time.monotonic()
            if now - last_dump >= args.checkpoint_interval:
                write_checkpoint(
                    checkpoint_path, out_file, normal_stack, full_stack, inflight_items
                )
                last_dump = now
                print(f"checkpointed ({written} written so far)", file=sys.stderr)

            timeout = min(60.0, max(1.0, args.checkpoint_interval - (now - last_dump)))
            try:
                batch_id, results = result_queue.get(timeout=timeout)
            except queue.Empty:
                continue

            inflight_items.pop(batch_id, None)
            for line, normal_children, full_children in results:
                out_file.write(line + "\n")
                written += 1
                normal_stack.extend(normal_children)  # push -> popped next == DFS
                full_stack.extend(full_children)
            if written % 10000 < len(results):
                print(
                    f"wrote {written} folders, "
                    f"stacks {len(normal_stack)}/{len(full_stack)}",
                    file=sys.stderr,
                )
            dispatch()
    finally:
        for _ in procs:
            task_queue.put(None)  # sentinels
        for p in procs:
            p.join()
        out_file.flush()
        out_file.close()

    # Scan finished cleanly: drop the checkpoint so a rerun starts fresh.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"done, wrote {written} folders to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
