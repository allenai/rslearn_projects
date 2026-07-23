"""Compute accuracy metrics for a category set against hand-labeled ground truth.

Reads the category set produced by ``run_gemini`` and, for points that have at least one
ground-truth change category, reports two metrics:

- Exact accuracy: the (pre, post, same) change categories all match exactly (including
  when both are null).
- Mostly-right accuracy: at least one non-null change category matches in the same slot.
  For a same-class ground truth, the only non-null slot is ``same``, so this reduces to
  getting it exactly right; for a pre/post ground truth it means we got at least one of
  the two sides correct.

Both metrics are also reported a second time excluding points the model flagged for
review, to show accuracy on just the points it was willing to categorize.

Points without any ground-truth change category are skipped.

Example:
    python -m rslp.change_finder_v2.vlm.category_tagger.compute_accuracy \
        --predictions categories.json
"""

from __future__ import annotations

import argparse

from .schema import CategorySet


def compute_metrics(category_set: CategorySet) -> dict[str, float | int]:
    """Compute exact and mostly-right accuracy over the labeled subset."""
    n = 0
    skipped = 0
    exact = 0
    mostly = 0
    flagged = 0
    n_unflagged = 0
    exact_unflagged = 0
    mostly_unflagged = 0
    pre_agree = 0
    post_agree = 0
    same_agree = 0

    for item in category_set.predictions:
        record = item.record
        gt = (
            record.gt_pre_change_category,
            record.gt_post_change_category,
            record.gt_same_change_category,
        )
        if not any(gt):
            skipped += 1
            continue
        pred = (
            item.pre_change_category,
            item.post_change_category,
            item.same_change_category,
        )

        n += 1
        is_exact = gt == pred
        if is_exact:
            exact += 1

        slot_agree = [g is not None and g == p for g, p in zip(gt, pred)]
        is_mostly = any(slot_agree)
        if is_mostly:
            mostly += 1
        pre_agree += int(slot_agree[0])
        post_agree += int(slot_agree[1])
        same_agree += int(slot_agree[2])

        if item.flagged_for_review:
            flagged += 1
        else:
            n_unflagged += 1
            exact_unflagged += int(is_exact)
            mostly_unflagged += int(is_mostly)

    return {
        "n": n,
        "skipped": skipped,
        "exact": exact,
        "mostly": mostly,
        "flagged": flagged,
        "exact_accuracy": exact / n if n else 0.0,
        "mostly_accuracy": mostly / n if n else 0.0,
        "n_unflagged": n_unflagged,
        "exact_unflagged": exact_unflagged,
        "mostly_unflagged": mostly_unflagged,
        "exact_accuracy_unflagged": exact_unflagged / n_unflagged if n_unflagged else 0.0,
        "mostly_accuracy_unflagged": (
            mostly_unflagged / n_unflagged if n_unflagged else 0.0
        ),
        "pre_agree": pre_agree,
        "post_agree": post_agree,
        "same_agree": same_agree,
    }


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compute accuracy metrics for a category set against ground truth."
    )
    parser.add_argument(
        "--predictions", required=True, help="Category set JSON from run_gemini."
    )
    parsed = parser.parse_args(args=args)

    category_set = CategorySet.load(parsed.predictions)
    metrics = compute_metrics(category_set)

    print(f"Labeled points:     {metrics['n']} (skipped {metrics['skipped']})")
    print(
        f"Exact accuracy:     {metrics['exact_accuracy']:.3f} "
        f"({metrics['exact']}/{metrics['n']})"
    )
    print(
        f"Mostly-right:       {metrics['mostly_accuracy']:.3f} "
        f"({metrics['mostly']}/{metrics['n']})"
    )
    print(f"Flagged for review: {metrics['flagged']}/{metrics['n']}")
    print(f"Excluding flagged points ({metrics['n_unflagged']} points):")
    print(
        f"  Exact accuracy:   {metrics['exact_accuracy_unflagged']:.3f} "
        f"({metrics['exact_unflagged']}/{metrics['n_unflagged']})"
    )
    print(
        f"  Mostly-right:     {metrics['mostly_accuracy_unflagged']:.3f} "
        f"({metrics['mostly_unflagged']}/{metrics['n_unflagged']})"
    )
    print("Per-slot agreement (non-null matches):")
    print(f"  pre:  {metrics['pre_agree']}")
    print(f"  post: {metrics['post_agree']}")
    print(f"  same: {metrics['same_agree']}")


if __name__ == "__main__":
    main()
