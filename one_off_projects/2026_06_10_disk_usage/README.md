# Disk usage scanner + viewer

A three-stage pipeline for figuring out what is eating disk on a huge filesystem
(e.g. weka), built to stay bounded in memory/output even on trees with millions
of directories and very high fan-out.

1. `disk_usage.py` — multiprocess scan that walks the tree and writes one JSONL
   line per directory (direct files only).
2. `collapse.py` — single streaming pass that folds the JSONL into a bounded
   nested tree JSON.
3. `app.py` — tiny Flask app that serves the collapsed tree as an interactive,
   expandable tree list (like ncdu) in the browser.

Stages are decoupled on purpose: the scan is the only slow/filesystem-bound part,
so you run it once and then re-collapse / re-view with different thresholds
without rescanning.

## 1. Scan the filesystem

```bash
python one_off_projects/2026_06_10_disk_usage/disk_usage.py \
    --root /weka/dfive-default \
    --output disk_usage.jsonl \
    --workers 64
```

How it stays bounded:

- The main process owns a DFS *stack* of pending directories, so the in-memory
  frontier stays ~depth x branching instead of a full BFS level.
- Directories are dispatched/returned in *batches* (`--batch_size`, default 100)
  to amortize IPC overhead, with an in-flight cap (`--max_inflight`) for
  backpressure.
- Dynamic fan-out collapse: a folder with more than `--fanout_threshold` (1000)
  subfolders has each subfolder accounted as a single recursive total instead of
  being recursed into, bounding output size on high-fan-out trees.

Resumability: every `--checkpoint_interval` seconds (default 1800) the pending
frontier is written atomically to `<output>.ckpt`. If that file exists at
startup the scan resumes and appends to the JSONL (duplicates are harmless;
`collapse.py` dedups by path). The checkpoint is removed on clean completion.

### Run the scan on Beaker

The scan is the only slow, filesystem-bound stage, so it's the part worth running
on a Beaker host that mounts weka directly. Use the common `beaker_launcher`
workflow. Build a Beaker image that includes `rslearn_projects` per the
instructions in `rslp/olmoearth_pretrain/README.md`, then assume the image (e.g.
`YOUR_BEAKER_IMAGE`) is available.

No GPU is needed. Mount the weka bucket and write the JSONL (and its checkpoint)
to the mount so the output survives the job, then point `--command` at the scan
script (the repo is the image's working directory):

```bash
python -m rslp.main common beaker_launcher \
    --image YOUR_BEAKER_IMAGE \
    --clusters '["ai2/jupiter"]' \
    --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' \
    --command '["python", "one_off_projects/2026_06_10_disk_usage/disk_usage.py", "--root", "/weka/dfive-default", "--output", "/weka/dfive-default/tmp/disk_usage.jsonl", "--workers", "64"]'
```

Since the JSONL lands on weka, the checkpoint does too, so a preempted job resumes
from `<output>.ckpt` on its next run. Afterwards run the cheap `collapse.py` and
`app.py` stages locally against the produced JSONL.

## 2. Collapse into a bounded tree

```bash
python one_off_projects/2026_06_10_disk_usage/collapse.py \
    --input disk_usage.jsonl \
    --output collapsed.json \
    --max_depth 10 \
    --collapse_gb 10 \
    --max_children 100
```

- `--max_depth`: directories deeper than this (relative to the scan root) are
  folded into their depth-`max_depth` ancestor, which becomes a leaf carrying the
  subtree total.
- `--collapse_gb`: folders whose total size is below this (GiB) are emitted
  without their children (aggregate size kept) for a lean payload.
- `--max_children`: folders with more than this many direct children are likewise
  emitted without their children.

Re-run with different thresholds to trade detail vs. payload size without
rescanning.

## 3. View in the browser

```bash
python one_off_projects/2026_06_10_disk_usage/app.py \
    --input collapsed.json \
    --host 127.0.0.1 --port 5000
```

The JSON is loaded once at startup and served verbatim from memory, so page
loads are cheap. Open http://127.0.0.1:5000 to explore the tree.
