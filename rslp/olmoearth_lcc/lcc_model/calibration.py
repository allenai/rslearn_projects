"""Calibrated change scores of the LCC model.

The model's softmax outputs are mostly near 0 or 1, so the stored scores are
remapped to estimates of precision, fitted on the validation set (see
``fit_score_calibration``):

- binary change: P(any change) = sigmoid(a * x + b) with
  x = log10(p_change / p_no_change).
- pre/post change category, where the head's argmax is a category:
  P(exact category) = sigmoid(a * x + b) * sigmoid(c * m + d) with
  x = log10(p_top_category / p_none) and
  m = log10(p_top_category / p_second_category) (categories only, none excluded).
- pre/post change category, where the head's argmax is none:
  P(none is correct) = 1 - sigmoid(a * x + b).

Each head only uses its own outputs, so the binary and category scores can be
inconsistent. x and m are clipped to [-X_CLIP, X_CLIP], the range the validation
fit saw (its float16 probabilities were clipped at PROB_EPS).

The curves are fit on all validation points of a head with a Platt fit, then
shifted for deployment: assuming the deployment change rate equals the rate of
predicted changes (so precision at the argmax threshold equals recall R), the
shift is -logit(pi_val) + log(FPR_val) - log(1 - R), where pi_val is the
validation positive rate and FPR_val the validation false positive rate at the
argmax threshold. The margin curve is a Platt fit of "category exact" on the
validation positives with a category predicted; with fewer than
MIN_MARGIN_ERRORS wrong categories there, it is the constant exact rate (a = 0).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from .output_bands import NONE_CATEGORY_IDX

# Probabilities are clipped to this before taking ratios (the validation fit used
# float16 probabilities, whose smallest subnormal is ~6e-8).
PROB_EPS = 6e-8
# x and m are clipped to this range (log10(1 / PROB_EPS) ~ 7.2).
X_CLIP = 7.2
# Below this many wrong categories, the margin curve is constant (too few errors to
# fit a slope; e.g. the pre head is ~99% exact).
MIN_MARGIN_ERRORS = 10


@dataclass(frozen=True)
class Curve:
    """A logistic curve sigmoid(a * x + b)."""

    a: float
    b: float


@dataclass(frozen=True)
class ScoreCalibration:
    """Calibration of the binary and change-category scores.

    Attributes:
        binary: curve of P(any change) on the binary head's x.
        pre: curve of P(pre change category) on the pre head's x.
        post: curve of P(post change category) on the post head's x.
        pre_margin: curve of P(exact | pre change) on the pre head's margin.
        post_margin: curve of P(exact | post change) on the post head's margin.
        source: free-form description of the fit (checkpoint, date).
    """

    binary: Curve
    pre: Curve
    post: Curve
    pre_margin: Curve
    post_margin: Curve
    source: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoreCalibration:
        """Parse a calibration from its dict form (as in the predict config).

        Args:
            d: map with keys binary, pre, post, pre_margin, post_margin (each a
                dict with a and b) and optionally source.

        Returns:
            the calibration.
        """
        curves = {
            k: Curve(float(d[k]["a"]), float(d[k]["b"]))
            for k in ("binary", "pre", "post", "pre_margin", "post_margin")
        }
        return cls(**curves, source=str(d.get("source", "")))

    def to_dict(self) -> dict[str, Any]:
        """Get the dict form of this calibration."""
        return asdict(self)


def _sigmoid_curve(curve: Curve, x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(curve.a * x + curve.b)


def _log10_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    ratio = num.clamp(min=PROB_EPS) / den.clamp(min=PROB_EPS)
    return torch.log10(ratio).clamp(-X_CLIP, X_CLIP)


def quantize(probs: torch.Tensor) -> torch.Tensor:
    """Scale probabilities to 0..255 as stored in the output raster."""
    return (probs * 255).clamp(0, 255).round()


def binary_score(
    probs: torch.Tensor, calibration: ScoreCalibration | None
) -> torch.Tensor:
    """Get the binary change score.

    Args:
        probs: (3, H, W) binary head softmax (nodata, no change, change).
        calibration: the calibration, or None for the raw change probability.

    Returns:
        the (H, W) score in [0, 1].
    """
    if calibration is None:
        return probs[2]
    return _sigmoid_curve(calibration.binary, _log10_ratio(probs[2], probs[1]))


def category_score(
    probs: torch.Tensor, curve: Curve | None, margin_curve: Curve | None
) -> torch.Tensor:
    """Get the score of a change-category head's argmax class.

    The argmax skips the nodata class and is taken over the quantized
    probabilities, as in the GeoZarr conversion, so that the score matches the
    stored category even where probabilities tie after quantization.

    Args:
        probs: (C, H, W) head softmax ([nodata, none, categories...]).
        curve: the head's curve, or None for the raw argmax probability.
        margin_curve: the head's margin curve (used with curve).

    Returns:
        the (H, W) score in [0, 1]: P(exact category) where the argmax is a
        category, P(none is correct) where it is none.
    """
    argmax = quantize(probs)[1:].argmax(dim=0) + 1
    if curve is None or margin_curve is None:
        return torch.gather(probs, 0, argmax[None])[0]
    top2 = probs[NONE_CATEGORY_IDX + 1 :].topk(2, dim=0).values
    x = _log10_ratio(top2[0], probs[NONE_CATEGORY_IDX])
    margin = _log10_ratio(top2[0], top2[1])
    p_change = _sigmoid_curve(curve, x)
    exact = p_change * _sigmoid_curve(margin_curve, margin)
    return torch.where(argmax == NONE_CATEGORY_IDX, 1 - p_change, exact)


def score_bands(
    binary: torch.Tensor,
    pre_change: torch.Tensor,
    post_change: torch.Tensor,
    calibration: ScoreCalibration | None,
) -> torch.Tensor:
    """Get the three score bands (binary, pre, post) in [0, 1].

    Args:
        binary: (3, H, W) binary head softmax.
        pre_change: (C_pre, H, W) pre change head softmax.
        post_change: (C_post, H, W) post change head softmax.
        calibration: the calibration, or None for the raw probabilities (binary
            change probability and each category head's argmax probability).

    Returns:
        the (3, H, W) scores.
    """
    return torch.stack(
        [
            binary_score(binary, calibration),
            category_score(
                pre_change,
                calibration.pre if calibration else None,
                calibration.pre_margin if calibration else None,
            ),
            category_score(
                post_change,
                calibration.post if calibration else None,
                calibration.post_margin if calibration else None,
            ),
        ]
    )


def read_score_calibration(model_config: str) -> dict[str, Any] | None:
    """Read the score calibration of a predict model config.

    Args:
        model_config: path of the model config (LCCMultiTask task).

    Returns:
        the task's score_calibration init arg, or None if it has none.
    """
    import yaml

    with open(model_config) as f:
        cfg = yaml.safe_load(f)
    task_args = cfg["data"]["init_args"]["task"].get("init_args", {})
    return task_args.get("score_calibration")


def _platt(x: npt.NDArray, y: npt.NDArray) -> Curve:
    """Fit P(y | x) = sigmoid(a * x + b) by (effectively unregularized) logistic regression."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e4).fit(x[:, None], y.astype(int))
    return Curve(float(model.coef_[0, 0]), float(model.intercept_[0]))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def fit_score_calibration(
    probs_binary: npt.NDArray,
    probs_pre_change: npt.NDArray,
    probs_post_change: npt.NDArray,
    binary: npt.NDArray,
    pre_change: npt.NDArray,
    post_change: npt.NDArray,
    pre_change_valid: npt.NDArray,
    post_change_valid: npt.NDArray,
    source: str = "",
) -> tuple[ScoreCalibration, dict[str, Any]]:
    """Fit the score calibration on validation points.

    One point per validation window (the labeled point). Windows kept for the
    category heads: labeled no change, plus changes with a pre or post category
    (changes labeled without any category are dropped); the binary head keeps
    every labeled window.

    Args:
        probs_binary: (N, 3) binary softmax.
        probs_pre_change: (N, C_pre) pre change softmax.
        probs_post_change: (N, C_post) post change softmax.
        binary: (N,) binary label (0 = unlabeled, 1 = no change, 2 = change).
        pre_change: (N,) pre change category label.
        post_change: (N,) post change category label.
        pre_change_valid: (N,) whether pre_change is labeled.
        post_change_valid: (N,) whether post_change is labeled.
        source: description stored in the calibration.

    Returns:
        the calibration and a dict of fit statistics.
    """
    stats: dict[str, Any] = {}
    curves: dict[str, Curve] = {}
    cat_labeled = (post_change_valid > 0) & ((post_change > 1) | (pre_change > 1))
    head_inputs = {
        "binary": (probs_binary, binary == 2, binary > 0),
        "pre": (
            probs_pre_change,
            np.where(pre_change_valid > 0, pre_change, 0),
            (binary == 1) | cat_labeled,
        ),
        "post": (
            probs_post_change,
            np.where(post_change_valid > 0, post_change, 0),
            (binary == 1) | cat_labeled,
        ),
    }
    for head, (probs, labels, keep) in head_inputs.items():
        p = np.clip(probs[keep].astype(np.float64), PROB_EPS, None)
        if head == "binary":
            positive = labels[keep]
            x = np.log10(p[:, 2] / p[:, 1])
        else:
            lab = labels[keep]
            positive = lab > NONE_CATEGORY_IDX
            cats = np.sort(p[:, NONE_CATEGORY_IDX + 1 :], axis=1)
            x = np.log10(cats[:, -1] / p[:, NONE_CATEGORY_IDX])
            margin = np.log10(cats[:, -1] / cats[:, -2])
            argmax = p[:, 1:].argmax(axis=1) + 1
            predicted = positive & (argmax > NONE_CATEGORY_IDX)
            exact = argmax[predicted] == lab[predicted]
            if (~exact).sum() < MIN_MARGIN_ERRORS:
                curves[f"{head}_margin"] = Curve(0.0, _logit(float(exact.mean())))
            else:
                curves[f"{head}_margin"] = _platt(
                    np.clip(margin[predicted], -X_CLIP, X_CLIP), exact
                )
            stats[f"{head}_margin"] = dict(
                n=int(predicted.sum()), n_exact=int(exact.sum())
            )
        x = np.clip(x, -X_CLIP, X_CLIP)
        val_curve = _platt(x, positive)
        pi_val = float(positive.mean())
        recall = float((x[positive] > 0).mean())
        fpr_val = float((x[~positive] > 0).mean())
        shift = -_logit(pi_val) + math.log(fpr_val) - math.log(1 - recall)
        curves[head] = Curve(val_curve.a, val_curve.b + shift)
        stats[head] = dict(
            n=int(keep.sum()),
            n_pos=int(positive.sum()),
            pi_val=pi_val,
            recall=recall,
            fpr_val=fpr_val,
            val_curve=asdict(val_curve),
            shift=shift,
        )
    return ScoreCalibration(**curves, source=source), stats
