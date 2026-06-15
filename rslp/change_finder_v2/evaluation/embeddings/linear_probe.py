"""Score change from embeddings via a supervised linear probe.

Fits a logistic-regression probe on the labeled training points (group ``train``,
windows ``train_{idx:06d}_src`` / ``_dst``) using the elementwise absolute difference of
the src/dst embeddings as features, then scores the eval points (group ``predict``,
windows ``eval_{idx:06d}_src`` / ``_dst``). ``change_score`` is the predicted P(change).

Both the train and eval points must already be embedded in the dataset; create them with
``create_prediction_dataset_from_csv.py --csv eval.csv --train-csv train.csv`` where
train.csv comes from running export_annotations_to_csv.py on the training v2 JSONs.

Works for both AlphaEarth (64-d) and OlmoEarth (768-d) embeddings; the probe is fit to
whatever embedding dimension is present.

    python -m rslp.change_finder_v2.evaluation.embeddings.linear_probe \
        --csv eval.csv --train-csv train.csv --ds-path "$EMBED_DS" \
        --output eval_alphaearth_probe.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from upath import UPath

from rslp.change_finder_v2.evaluation.embeddings.common import (
    PREDICTION_GROUP,
    TRAIN_GROUP,
    base_row,
    iter_points_with_embeddings,
    load_points,
    read_embedding_at_point,
    write_merged_csv,
)


def _feature(e_src: np.ndarray, e_dst: np.ndarray) -> np.ndarray:
    """Linear-probe feature: elementwise absolute difference of the embeddings."""
    return np.abs(e_src - e_dst)


def _build_training_matrix(
    ds_upath: UPath, train_csv: Path
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build (X_train, y_train, num_used, num_total) from the training group."""
    points = load_points(train_csv)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for point, e_src, e_dst in iter_points_with_embeddings(
        ds_upath, points, TRAIN_GROUP, "train_"
    ):
        features.append(_feature(e_src, e_dst))
        labels.append(1 if point.has_changed else 0)
    if not features:
        raise ValueError(
            "no training points had both src and dst embeddings materialized"
        )
    return np.stack(features), np.array(labels), len(features), len(points)


def linear_probe(
    csv_path: Path, train_csv_path: Path, ds_path: str, output: Path
) -> list[dict[str, Any]]:
    """Fit the probe on the train group and score the eval group into a merged CSV."""
    ds_upath = UPath(ds_path)

    x_train, y_train, used, total = _build_training_matrix(ds_upath, train_csv_path)
    print(
        f"Training probe on {used}/{total} train points "
        f"({int(y_train.sum())} changed, {int((1 - y_train).sum())} no-change), "
        f"feature dim {x_train.shape[1]}"
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(x_train, y_train)

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
            feat = _feature(e_src, e_dst).reshape(1, -1)
            prob = float(clf.predict_proba(feat)[0, 1])
            out["change_score"] = round(prob, 6)
            out["predicted_changed"] = bool(prob >= 0.5)
            out["has_prediction"] = True
        merged.append(out)

    write_merged_csv(merged, output)

    scored = sum(1 for r in merged if r["has_prediction"])
    missing = len(merged) - scored
    print(
        f"Wrote {len(merged)} rows to {output} (linear_probe); "
        f"{scored} with predictions, {missing} missing (no src and/or dst embedding)"
    )
    return merged


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Score embedding change with a supervised logistic-regression probe."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Evaluation CSV (eval points, group 'predict').",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="Training CSV (labeled points, group 'train') used to fit the probe.",
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
    args = parser.parse_args()

    linear_probe(
        csv_path=args.csv,
        train_csv_path=args.train_csv,
        ds_path=args.ds_path,
        output=args.output,
    )


if __name__ == "__main__":
    main()
