# Weka HDD -> SSD cache warmer

`warm_files.py` pulls a directory tree from the weka HDD tier into the SSD cache
by simply reading every file under a root (reading a file's bytes promotes it
into the SSD cache). The bytes are read in chunks and discarded — the read
itself is the point.

It's a stripped-down sibling of `../2026_06_10_disk_usage/disk_usage.py`: same
multiprocess DFS-stack + batched-dispatch structure, but no fan-out collapse, no
checkpointing, and no resumability. It's a one-shot warm-up that only reports
aggregate stats to stderr.

## Run

```bash
python one_off_projects/2026_06_24_weka_hdd_to_ssd/warm_files.py \
    --root /weka/dfive-default/some/dataset \
    --workers 128
```

Options:

- `--workers` (128): number of worker processes.
- `--batch_size` (100): directories dispatched per task message (amortizes IPC).
- `--max_inflight` (0 -> 2 x workers): max batches dispatched but not yet
  returned, providing backpressure so the queues stay bounded.
- `--chunk_size` (4 MiB): read size per chunk when reading each file.

Subdirectories are shuffled before being handed back, so launching the warmer on
several hosts at once traverses the tree in different orders and avoids all hosts
hammering the same files simultaneously — each run still eventually reads
everything.

## Run on Beaker

Use the common `beaker_launcher` workflow to run this on a Beaker host that mounts
the weka tier. Build a Beaker image that includes `rslearn_projects` per the
instructions in `rslp/olmoearth_pretrain/README.md`, then assume the image (e.g.
`YOUR_BEAKER_IMAGE`) is available.

The warmer just reads files, so no GPU is needed. Mount the weka bucket and point
`--command` at this script (the repo is the image's working directory):

```bash
python -m rslp.main common beaker_launcher \
    --image YOUR_BEAKER_IMAGE \
    --clusters '["ai2/jupiter"]' \
    --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' \
    --command '["python", "one_off_projects/2026_06_24_weka_hdd_to_ssd/warm_files.py", "--root", "/weka/dfive-default/some/dataset", "--workers", "128"]'
```

Because subdirectories are shuffled, you can launch this on several hosts at once
(re-run the command, optionally pinning a host with `--hostname` instead of
`--clusters`) to warm the tree faster without all hosts hammering the same files.
