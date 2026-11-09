"""Apply the final label rules to dataset_v1/20261103 (the 2026-11-07 label version).

Each original label (from the source dataset, 20250521) is matched by its window pixel
(row, col), and every label at the same position gets the same decision:

1. If favyen labeled the position in the review app, his label wins: vessel/moored keep,
   infra/island/derelict drop.
2. A label already dropped from the target dataset (earlier cleanup rounds) is restored
   only if Gemini says "correct" (--gemini_dropped).
3. A label still in the target dataset is dropped only if Gemini says "wrong"
   (--gemini_kept); correct, low_confidence and no answer keep it.

Window label files of the target dataset (layers/label/data.geojson) are replaced, never
edited in place. That matters because in 20261103 they started as hardlinks to 20250521.
The source label files are checked to be unchanged.

The script also writes three GeoJSONs to the target dataset root, with one point per label
(WGS84) and properties window, split, row and col:

    labels_orig.geojson      every label of the source dataset
    labels_<before>.geojson  the target dataset's labels before this script
    labels_<after>.geojson   the result, with the rule applied and the Gemini answer

Usage:

    python apply_labels.py apply --src_ds .../20250521 --ds .../20261103 \
        --app_dir review_app_dir [--app_dir ...] \
        --gemini_dropped dropped/results.json --gemini_kept kept/results.json \
        --before_name labels_20261105 --after_name labels_20261107 [--dry_run]
    python apply_labels.py verify --src_ds ... --ds ... --after_name labels_20261107
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool

from rasterio.warp import transform

FAVYEN_DECISION = {
    "infra": "drop",
    "island": "drop",
    "derelict": "drop",
    "vessel": "keep",
    "moored": "keep",
}

Position = tuple[str, int, int]


def sha256(path: str) -> str:
    """SHA-256 of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_app_labels(app_dirs: list[str]) -> dict[Position, dict]:
    """favyen's review-app labels by position; the latest one wins."""
    out: dict[Position, dict] = {}
    for app_dir in app_dirs:
        with open(f"{app_dir}/items.json") as f:
            items = {x["id"]: x for x in json.load(f)}
        with open(f"{app_dir}/labels.json") as f:
            labels = json.load(f)
        for item_id, v in labels.items():
            if item_id not in items or v["label"] not in FAVYEN_DECISION:
                continue
            x = items[item_id]
            pos = (x["win"], x["row"], x["col"])
            prev = out.get(pos)
            if prev is None or (v.get("at") or "") >= (prev["at"] or ""):
                conflict = prev is not None and (
                    FAVYEN_DECISION[prev["label"]] != FAVYEN_DECISION[v["label"]]
                )
                out[pos] = {"label": v["label"], "at": v.get("at"), "conflict": conflict}
    return out


def load_gemini(path: str) -> dict[Position, dict]:
    """Gemini answers ({answer, category, reason}) by position, from a results.json."""
    with open(path) as f:
        results = json.load(f)
    return {
        (r["win"], r["row"], r["col"]): r["answer"]
        for r in results.values()
        if "answer" in r
    }


def window_pos(meta: dict, feat: dict) -> tuple[int, int]:
    """(row, col) of a label within its window."""
    x0, y0 = meta["bounds"][:2]
    x, y = feat["geometry"]["coordinates"][:2]
    return int(round(y - y0)), int(round(x - x0))


def process_window(
    args: argparse.Namespace,
    app: dict[Position, dict],
    g_dropped: dict[Position, dict],
    g_kept: dict[Position, dict],
    win: str,
) -> dict:
    """Apply the rules to one window."""
    src_dir, dst_dir = f"{args.src_ds}/windows/{win}", f"{args.ds}/windows/{win}"
    with open(f"{src_dir}/metadata.json") as f:
        meta = json.load(f)
    p_src = f"{src_dir}/layers/label/data.geojson"
    p_dst = f"{dst_dir}/layers/label/data.geojson"
    with open(p_src) as f:
        fc_src = json.load(f)
    with open(p_dst) as f:
        fc_dst = json.load(f)
    cur_pos = {window_pos(meta, feat) for feat in fc_dst["features"]}
    split = "val" if "-val_" in win else "train"

    feats = fc_src["features"]
    proj = meta["projection"]
    xs = [f["geometry"]["coordinates"][0] * proj["x_resolution"] for f in feats]
    ys = [f["geometry"]["coordinates"][1] * proj["y_resolution"] for f in feats]
    lons, lats = transform(proj["crs"], "EPSG:4326", xs, ys) if xs else ([], [])

    keep, out_orig, out_before, out_after = [], [], [], []
    rules: Counter = Counter()
    for feat, lon, lat in zip(feats, lons, lats):
        row, col = window_pos(meta, feat)
        pos = (win, row, col)
        in_cur = (row, col) in cur_pos
        fav = app.get(pos)
        gem = (g_kept if in_cur else g_dropped).get(pos)
        if fav:
            decision, rule = FAVYEN_DECISION[fav["label"]], f"favyen_{fav['label']}"
        elif not in_cur:
            if gem and gem["answer"] == "correct":
                decision, rule = "keep", "restored_gemini_correct"
            else:
                decision, rule = "drop", "stays_dropped"
        elif gem and gem["answer"] == "wrong":
            decision, rule = "drop", "gemini_wrong"
        else:
            decision, rule = "keep", f"kept_gemini_{gem['answer']}" if gem else "kept_no_answer"
        state = "in_before" if in_cur else "dropped_before"
        rules[f"{split} | {state} | {rule} | {decision}"] += 1

        def point(**props: object) -> dict:
            return {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": dict(
                    feat["properties"], window=win, split=split, row=row, col=col, **props
                ),
            }

        out_orig.append(point())
        if in_cur:
            out_before.append(point())
        if decision == "keep":
            keep.append(feat)
            props: dict = {"rule": rule}
            if fav:
                props["favyen_label"] = fav["label"]
            if gem:
                props.update(
                    gemini_answer=gem["answer"],
                    gemini_category=gem.get("category"),
                    gemini_reason=gem.get("reason"),
                )
            out_after.append(point(**props))

    def key(feat: dict) -> str:
        return json.dumps(feat, sort_keys=True)

    changed = sorted(map(key, keep)) != sorted(map(key, fc_dst["features"]))
    if changed and not args.dry_run:
        h_src = sha256(p_src)
        with open(p_dst + ".tmp", "w") as f:
            json.dump(dict(fc_dst, features=keep), f)
        os.replace(p_dst + ".tmp", p_dst)
        assert sha256(p_src) == h_src, f"source label file changed: {win}"
        assert os.stat(p_dst).st_ino != os.stat(p_src).st_ino
    return {
        "orig": out_orig,
        "before": out_before,
        "after": out_after,
        "rules": rules,
        "changed": changed,
        "n_before": len(fc_dst["features"]),
        "n_after": len(keep),
    }


def list_windows(ds: str) -> list[str]:
    """'group/name' of every window."""
    return sorted(
        f"{g}/{w}" for g in os.listdir(f"{ds}/windows") for w in os.listdir(f"{ds}/windows/{g}")
    )


def apply(args: argparse.Namespace) -> None:
    """Apply the rules to every window and write the root GeoJSONs."""
    app = load_app_labels(args.app_dir)
    conflicts = [pos for pos, v in app.items() if v["conflict"]]
    assert not conflicts, f"review-app labels disagree at {conflicts[:5]}"
    g_dropped = load_gemini(args.gemini_dropped)
    g_kept = load_gemini(args.gemini_kept)
    wins = list_windows(args.src_ds)
    fn = partial(process_window, args, app, g_dropped, g_kept)
    with Pool(args.workers) as p:
        res = p.map(fn, wins, chunksize=64)
    out_dir = args.out_dir if args.dry_run else args.ds
    for name, k in (
        ("labels_orig", "orig"),
        (args.before_name, "before"),
        (args.after_name, "after"),
    ):
        fc = {"type": "FeatureCollection", "features": [f for r in res for f in r[k]]}
        dst = f"{out_dir}/{name}.geojson"
        with open(dst + ".tmp", "w") as f:
            json.dump(fc, f)
        os.replace(dst + ".tmp", dst)
        print(name, len(fc["features"]), "->", dst)
    rules = sum((r["rules"] for r in res), Counter())
    summary = {
        "dry_run": args.dry_run,
        "windows": len(wins),
        "windows_changed": sum(r["changed"] for r in res),
        "labels_before": sum(r["n_before"] for r in res),
        "labels_after": sum(r["n_after"] for r in res),
        "rules": dict(sorted(rules.items())),
    }
    print(json.dumps(summary, indent=1))


def window_label_positions(ds: str, win: str) -> tuple[str, list[tuple[int, int]] | None]:
    """Sorted (row, col) of the labels in a window's label file."""
    path = f"{ds}/windows/{win}/layers/label/data.geojson"
    if not os.path.exists(path):
        return win, None
    with open(f"{ds}/windows/{win}/metadata.json") as f:
        meta = json.load(f)
    with open(path) as f:
        fc = json.load(f)
    return win, sorted(window_pos(meta, feat) for feat in fc["features"])


def verify(args: argparse.Namespace) -> None:
    """Check window label files against the root GeoJSONs.

    The target dataset should match labels_<after> and the source dataset labels_orig.
    """
    wins = list_windows(args.ds)
    with Pool(args.workers) as p:
        cur = dict(p.map(partial(window_label_positions, args.ds), wins, chunksize=64))
        orig = dict(p.map(partial(window_label_positions, args.src_ds), wins, chunksize=64))
    for name, files in ((args.after_name, cur), ("labels_orig", orig)):
        ref: dict[str, list] = defaultdict(list)
        with open(f"{args.ds}/{name}.geojson") as f:
            for feat in json.load(f)["features"]:
                p = feat["properties"]
                ref[p["window"]].append((p["row"], p["col"]))
        bad = [w for w in wins if (files[w] or []) != sorted(ref.get(w, []))]
        print(name, "windows", len(wins), "mismatched", len(bad), bad[:5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the 2026-11-07 label rules.")
    parser.add_argument("command", choices=["apply", "verify"])
    parser.add_argument("--src_ds", required=True, help="source dataset (read only)")
    parser.add_argument("--ds", required=True, help="dataset whose labels are edited")
    parser.add_argument("--app_dir", action="append", default=[])
    parser.add_argument("--gemini_dropped")
    parser.add_argument("--gemini_kept")
    parser.add_argument("--before_name", default="labels_20261105")
    parser.add_argument("--after_name", default="labels_20261107")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_dir", default=".", help="where dry runs write GeoJSONs")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    {"apply": apply, "verify": verify}[args.command](args)
