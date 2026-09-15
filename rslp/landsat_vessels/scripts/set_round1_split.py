"""Set (or restore) the train/val/test split of the round-1 windows.

``create_round1_windows.py`` assigns a frozen scene-level split, stratified by
stratum x tier: train 1,306 / val 266 / test 428 detections over 114 / 21 / 60 scenes.
Use this to put every window in one split instead — e.g. all ``train``, to train on the
whole pool and evaluate on a different group entirely.

The split lives in three places, and this keeps them consistent:

- each window's ``metadata.json`` ``options.split`` — the authoritative one, since
  rslearn's ``SplitConfig.tags`` matches against window options;
- the annotation index, which the app reads for its split filter and drawer;
- the ``split`` property on any label layer already written, which is provenance only but
  should not contradict the window it sits on.

The original assignment is written to ``<group>_frozen_split.json`` before the first
overwrite and is never overwritten again, so ``--restore`` always recovers the frozen
split exactly. (It is also recomputable: the assignment is a deterministic sha256 of
scene id and stratum x tier cell.)

Usage:
    python set_round1_split.py --split train      # everything into one split
    python set_round1_split.py --restore          # put the frozen split back
    python set_round1_split.py --show             # report what is set now
"""

import argparse
import json
from collections import defaultdict

import tqdm
from rslearn.dataset import Dataset
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.landsat_vessels.annotations.writeback import LABEL_LAYER

DATASET_ROOT = UPath(
    "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624"
)
ROUND_DIR = UPath("/weka/dfive-default/yawenz/landsat/annotation_round1")
GROUP = "round1_20260803"


def summarize(windows: list, index: dict) -> None:
    """Print the split as it stands, from the windows themselves."""
    by_split: dict[str, int] = defaultdict(int)
    scenes: dict[str, set] = defaultdict(set)
    for window in windows:
        split = str(window.options.get("split"))
        by_split[split] += 1
        scenes[split].add(index[window.name]["scene_id"])
    print("  windows:", dict(sorted(by_split.items())))
    print("  scenes: ", {k: len(v) for k, v in sorted(scenes.items())})


def main() -> None:
    """Assign the train/val split tag for the round-1 window group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default=GROUP)
    parser.add_argument("--split", default=None, help="put every window in this split")
    parser.add_argument(
        "--restore", action="store_true", help="restore the frozen split"
    )
    parser.add_argument("--show", action="store_true", help="report and exit")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.split, args.restore, args.show)) != 1:
        parser.error("pass exactly one of --split, --restore, --show")

    index_path = ROUND_DIR / f"{args.group}_index.json"
    with index_path.open() as f:
        records = json.load(f)
    index = {record["window"]: record for record in records}

    dataset = Dataset(DATASET_ROOT)
    windows = dataset.load_windows(groups=[args.group])
    print(f"{len(windows)} windows in {args.group}")
    print("current:")
    summarize(windows, index)
    if args.show:
        return

    frozen_path = ROUND_DIR / f"{args.group}_frozen_split.json"
    if args.restore:
        if not frozen_path.exists():
            parser.error(f"no frozen split saved at {frozen_path}")
        with frozen_path.open() as f:
            target = json.load(f)
        print(f"\nrestoring the frozen split from {frozen_path}")
    else:
        # Save the frozen assignment once, before it is first overwritten.
        if not frozen_path.exists():
            with frozen_path.open("w") as f:
                json.dump({r["window"]: r["split"] for r in records}, f, indent=0)
            print(f"\nsaved the frozen split to {frozen_path}")
        else:
            print(f"\nfrozen split already saved at {frozen_path} (left as is)")
        target = {record["window"]: args.split for record in records}
        print(f"setting every window to split={args.split!r}")

    vector_format = GeojsonVectorFormat()
    changed = 0
    labels_updated = 0
    for window in tqdm.tqdm(windows, desc="updating windows"):
        want = target.get(window.name)
        if want is None:
            continue
        if window.options.get("split") != want:
            window.options["split"] = want
            window.save()
            changed += 1
        index[window.name]["split"] = want

        # Keep provenance on an already-written label layer from contradicting its window.
        if window.is_layer_completed(LABEL_LAYER):
            features = window.data.read_vector(LABEL_LAYER, vector_format)
            if features and features[0].properties.get("split") != want:
                features[0].properties["split"] = want
                with window.data.open_layer_writer(LABEL_LAYER) as writer:
                    writer.write_vector(vector_format, features)
                window.mark_layer_completed(LABEL_LAYER)
                labels_updated += 1

    with index_path.open("w") as f:
        json.dump([index[r["window"]] for r in records], f)

    print(
        f"\nrewrote {changed} window metadata, {labels_updated} label layers, and the index"
    )
    print("now:")
    summarize(dataset.load_windows(groups=[args.group]), index)
    print(
        "\nThe app reads the index at startup, so restart it for the UI's split filter to "
        "agree:\n"
        "  pgrep -af 'uvicorn rslp[.]land' | grep -v 'bash -c' | awk '{print $1}' | xargs -r kill"
    )


if __name__ == "__main__":
    main()
