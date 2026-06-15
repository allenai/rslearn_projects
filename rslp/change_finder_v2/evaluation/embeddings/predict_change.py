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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from upath import UPath

from rslp.change_finder_v2.evaluation.embeddings.common import (
    PREDICTION_GROUP,
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


def predict_change(
    csv_path: Path, ds_path: str, output: Path, method: str
) -> list[dict[str, Any]]:
    """Score each eval point by embedding distance and write the merged CSV."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choices: {sorted(METHODS)}")
    score_fn = METHODS[method]
    ds_upath = UPath(ds_path)

    points = load_points(csv_path)
    merged: list[dict[str, Any]] = []
    for point in points:
        out = base_row(point)
        e_src = read_embedding_at_point(
            ds_upath,
            PREDICTION_GROUP,
            f"eval_{point.row_index:06d}_src",
            point.lon,
            point.lat,
        )
        e_dst = read_embedding_at_point(
            ds_upath,
            PREDICTION_GROUP,
            f"eval_{point.row_index:06d}_dst",
            point.lon,
            point.lat,
        )
        if e_src is not None and e_dst is not None:
            out["change_score"] = round(score_fn(e_src, e_dst), 6)
            out["has_prediction"] = True
        merged.append(out)

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
    args = parser.parse_args()

    predict_change(
        csv_path=args.csv,
        ds_path=args.ds_path,
        output=args.output,
        method=args.method,
    )


if __name__ == "__main__":
    main()
