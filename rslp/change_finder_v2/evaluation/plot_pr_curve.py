"""Plot precision-recall curves for the change-detection methods.

Reads the per-method PR-curve CSVs written by ``metrics.py`` (columns
``threshold, precision, recall, f1, tp, fp, fn, tn``) from a directory and draws all
curves on a single, roughly-square, one-column figure intended to sit next to the
AUROC table in the paper.

Each method is matched by its CSV filename (see ``METHODS`` below). Missing files are
skipped, so the script still runs if only some methods are present.

    python -m rslp.change_finder_v2.evaluation.plot_pr_curve \
        --input-dir /tmp/b/ --output pr_curve.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (filename, legend label, color, linestyle, linewidth). Order controls legend order
# and draw order; the proposed method (LCMonitor) is drawn last/thickest so it sits on
# top. Labels match the AUROC table; color groups a method family and linestyle
# distinguishes its cosine-similarity vs. probe variants.
METHODS: list[tuple[str, str, str, str, float]] = [
    ("worldcover_pr.csv", "ESA WorldCover", "#9467bd", "-", 1.3),
    ("esri_io_pr.csv", "Esri Land Cover", "#8c564b", "-", 1.3),
    ("alphaearth_cosine_pr.csv", "AlphaEarth (cosine similarity)", "#ff7f0e", "--", 1.3),
    ("alphaearth_probe_pr.csv", "AlphaEarth (probe)", "#ff7f0e", ":", 1.6),
    ("olmoearth_embeddings_cosine_pr.csv", "OlmoEarth (cosine similarity)", "#1f77b4", "--", 1.3),  # noqa: E501
    ("olmoearth_embeddings_probe_concat_pr.csv", "OlmoEarth (probe)", "#1f77b4", ":", 1.6),  # noqa: E501
    ("lcc_model_pr.csv", "LCMonitor", "#2ca02c", "-", 2.2),
]


def _read_pr_curve(csv_path: Path) -> tuple[list[float], list[float]]:
    """Return (recall, precision) lists from a PR-curve CSV (in threshold order)."""
    recall: list[float] = []
    precision: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                p = float(row["precision"])
                r = float(row["recall"])
            except (KeyError, ValueError):
                continue
            # Skip undefined precision/recall (e.g. no positives past a threshold).
            if p != p or r != r:
                continue
            recall.append(r)
            precision.append(p)
    return recall, precision


def plot_pr_curves(input_dir: Path, output: Path) -> None:
    """Draw all available method PR curves to a single wide, short figure."""
    # Single-column width (~3.3in is a typical two-column paper column width), but
    # short so it does not take up much vertical space next to the AUROC table.
    fig, ax = plt.subplots(figsize=(3.3, 2.2))

    plotted = 0
    for filename, label, color, linestyle, linewidth in METHODS:
        csv_path = input_dir / filename
        if not csv_path.exists():
            print(f"Skipping missing {csv_path}")
            continue
        recall, precision = _read_pr_curve(csv_path)
        if not recall:
            print(f"Skipping empty {csv_path}")
            continue
        ax.plot(
            recall,
            precision,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit(f"No PR-curve CSVs found in {input_dir}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Let the plotting box fill the (wide, short) figure rather than forcing a square.
    ax.grid(True, linewidth=0.4, alpha=0.5)
    # Place the legend in two columns below the plot.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fontsize=6,
        frameon=False,
        borderpad=0.4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Wrote PR-curve figure with {plotted} methods to {output}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Plot precision-recall curves from per-method PR-curve CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the per-method PR-curve CSVs (e.g. /tmp/b/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pr_curve.pdf"),
        help="Output figure path (extension sets the format). Default: pr_curve.pdf.",
    )
    args = parser.parse_args()

    plot_pr_curves(input_dir=args.input_dir, output=args.output)


if __name__ == "__main__":
    main()
