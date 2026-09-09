"""Score change from embeddings via embedding distance (unsupervised, cosine mode).

For each eval point, read the src-year and dst-year embeddings (group ``predict``,
windows ``eval_{idx:06d}_src`` / ``_dst``) and compute a change score from the distance
between them. Works for both AlphaEarth (64-d) and OlmoEarth (768-d) embeddings.

The output CSV uses the shared standardized schema (category / predicted_changed columns
stay blank, since embeddings give no class), so a single metric script can consume it
alongside the WorldCover and linear-probe outputs.

    python -m rslp.change_finder_v2.evaluation.embeddings.predict_change \
        --csv eval.csv --ds-path "$EMBED_DS" --output eval_alphaearth_cosine.csv
"""

from __future__ import annotations

import argparse
import multiprocessing
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
from upath import UPath

from rslp.change_finder_v2.evaluation.embeddings.common import (
    EMBEDDINGS_LAYER,
    PREDICTION_GROUP,
    TRAIN_GROUP,
    PointRow,
    base_row,
    load_points,
    read_embedding_at_point,
    write_merged_csv,
)


def _cosine_distance(e_src: np.ndarray, e_dst: np.ndarray) -> float:
    """1 - cosine_similarity(e_src, e_dst), in [0, 2]; higher = more change."""
    denom = float(np.linalg.norm(e_src) * np.linalg.norm(e_dst))
    if denom == 0:
        return float("nan")
    return 1.0 - float(np.dot(e_src, e_dst)) / denom


# Registry of distance-based change-score methods; structured so more can be added.
METHODS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "cosine": _cosine_distance,
}
DEFAULT_METHOD = "cosine"

# Map each prediction group to the window-name prefix used when the dataset was created
# (see create_prediction_dataset_from_csv.py).
GROUP_NAME_PREFIXES: dict[str, str] = {
    PREDICTION_GROUP: "eval_",
    TRAIN_GROUP: "train_",
}

DEFAULT_WORKERS = 64


def _score_point(
    point: PointRow,
    ds_path: str,
    group: str,
    name_prefix: str,
    method: str,
    pre_layer: str,
    post_layer: str,
) -> dict[str, Any]:
    """Read src/dst embeddings for one point and build its merged-CSV row.

    Runs in a worker process, so it takes plain picklable args and reconstructs the
    UPath and score function locally.
    """
    score_fn = METHODS[method]
    ds_upath = UPath(ds_path)
    out = base_row(point)
    e_src = read_embedding_at_point(
        ds_upath,
        group,
        f"{name_prefix}{point.row_index:06d}_src",
        point.lon,
        point.lat,
        pre_layer,
    )
    e_dst = read_embedding_at_point(
        ds_upath,
        group,
        f"{name_prefix}{point.row_index:06d}_dst",
        point.lon,
        point.lat,
        post_layer,
    )
    if e_src is not None and e_dst is not None:
        out["change_score"] = round(score_fn(e_src, e_dst), 6)
        out["has_prediction"] = True
    return out


def predict_change(
    csv_path: Path,
    ds_path: str,
    output: Path,
    method: str,
    group: str = PREDICTION_GROUP,
    pre_layer: str = EMBEDDINGS_LAYER,
    post_layer: str = EMBEDDINGS_LAYER,
    workers: int = DEFAULT_WORKERS,
) -> list[dict[str, Any]]:
    """Score each eval point by embedding distance and write the merged CSV."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choices: {sorted(METHODS)}")
    if group not in GROUP_NAME_PREFIXES:
        raise ValueError(
            f"unknown group {group!r}; choices: {sorted(GROUP_NAME_PREFIXES)}"
        )
    name_prefix = GROUP_NAME_PREFIXES[group]

    points = load_points(csv_path)
    worker = partial(
        _score_point,
        ds_path=ds_path,
        group=group,
        name_prefix=name_prefix,
        method=method,
        pre_layer=pre_layer,
        post_layer=post_layer,
    )
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            merged = list(
                tqdm.tqdm(
                    pool.imap(worker, points),
                    total=len(points),
                    desc="Scoring points",
                )
            )
    else:
        merged = [
            worker(point)
            for point in tqdm.tqdm(points, desc="Scoring points")
        ]

    write_merged_csv(merged, output)

    scored = sum(1 for r in merged if r["has_prediction"])
    missing = len(merged) - scored
    print(
        f"Wrote {len(merged)} rows to {output} (method={method}); "
        f"{scored} with predictions, {missing} missing (no src and/or dst embedding)"
    )
    return merged


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Score embedding change (cosine distance) into a standardized CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Evaluation CSV used to create the prediction dataset.",
    )
    parser.add_argument(
        "--ds-path",
        required=True,
        help="Embeddings prediction dataset path (with materialized embeddings).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output method CSV path.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=sorted(METHODS),
        help=f"Distance method. Default: {DEFAULT_METHOD}.",
    )
    parser.add_argument(
        "--group",
        default=PREDICTION_GROUP,
        choices=sorted(GROUP_NAME_PREFIXES),
        help=(
            "Prediction dataset group to score. Use "
            f"{TRAIN_GROUP!r} to evaluate on the training points. "
            f"Default: {PREDICTION_GROUP}."
        ),
    )
    parser.add_argument(
        "--pre_layer",
        default=EMBEDDINGS_LAYER,
        help=f"Embeddings raster layer name for the pre (src) window. Default: {EMBEDDINGS_LAYER}.",
    )
    parser.add_argument(
        "--post_layer",
        default=EMBEDDINGS_LAYER,
        help=f"Embeddings raster layer name for the post (dst) window. Default: {EMBEDDINGS_LAYER}.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker processes for reading embeddings. Default: {DEFAULT_WORKERS}.",
    )
    args = parser.parse_args()

    predict_change(
        csv_path=args.csv,
        ds_path=args.ds_path,
        output=args.output,
        method=args.method,
        group=args.group,
        pre_layer=args.pre_layer,
        post_layer=args.post_layer,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
