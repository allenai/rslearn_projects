import math
import pathlib

import numpy as np
import pytest
import torch
import yaml

from rslp.olmoearth_lcc.lcc_model import calibration, geozarr
from rslp.olmoearth_lcc.lcc_model.calibration import Curve, ScoreCalibration
from rslp.olmoearth_lcc.lcc_model.output_bands import (
    NONE_CATEGORY_IDX,
    OUTPUT_BANDS,
    POST_CHANGE_BAND_OFFSET,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_BAND_OFFSET,
    PRE_CHANGE_CATEGORY_NAMES,
    SCORE_BAND_OFFSET,
)

CAL = ScoreCalibration(
    binary=Curve(0.8, -0.3),
    pre=Curve(0.5, -0.35),
    post=Curve(0.7, 0.0),
    pre_margin=Curve(0.4, 2.8),
    post_margin=Curve(0.55, -0.4),
    source="test",
)


def _sig(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def _probs(values: list[float]) -> torch.Tensor:
    """A (C, 1, 1) softmax from per-class probabilities."""
    return torch.tensor(values, dtype=torch.float32)[:, None, None]


def _post(none: float, cats: dict[str, float]) -> list[float]:
    """Post head probabilities with the given none and category probabilities."""
    p = [0.0] * len(POST_CHANGE_CATEGORY_NAMES)
    p[NONE_CATEGORY_IDX] = none
    for name, v in cats.items():
        p[POST_CHANGE_CATEGORY_NAMES.index(name)] = v
    return p


def test_dict_roundtrip() -> None:
    assert ScoreCalibration.from_dict(CAL.to_dict()) == CAL


def test_binary_score() -> None:
    probs = _probs([0.0, 0.01, 0.99])
    x = math.log10(0.99 / 0.01)
    assert calibration.binary_score(probs, CAL).item() == pytest.approx(
        _sig(0.8 * x - 0.3), abs=1e-5
    )
    assert calibration.binary_score(probs, None).item() == pytest.approx(0.99)


def test_category_score_category_branch() -> None:
    probs = _probs(_post(0.1, {"new_building": 0.8, "settlement": 0.1}))
    x = math.log10(0.8 / 0.1)
    m = math.log10(0.8 / 0.1)
    expected = _sig(0.7 * x) * _sig(0.55 * m - 0.4)
    score = calibration.category_score(probs, CAL.post, CAL.post_margin)
    assert score.item() == pytest.approx(expected, abs=1e-5)
    # Without a calibration: the argmax probability.
    assert calibration.category_score(probs, None, None).item() == pytest.approx(0.8)


def test_category_score_none_branch() -> None:
    probs = _probs(_post(0.9, {"new_building": 0.06, "settlement": 0.04}))
    x = math.log10(0.06 / 0.9)
    score = calibration.category_score(probs, CAL.post, CAL.post_margin)
    assert score.item() == pytest.approx(1 - _sig(0.7 * x), abs=1e-5)
    assert calibration.category_score(probs, None, None).item() == pytest.approx(0.9)


def test_category_score_quantized_tie_uses_none_branch() -> None:
    # none and the top category round to the same 0-255 value: the stored argmax
    # (first maximum over the quantized probabilities) is none, so the score must be
    # the none branch even though the float top category is slightly higher.
    probs = _probs(_post(0.4990, {"new_building": 0.4995, "settlement": 0.0015}))
    q = calibration.quantize(probs)
    assert q[NONE_CATEGORY_IDX] == q[POST_CHANGE_CATEGORY_NAMES.index("new_building")]
    x = math.log10(0.4995 / 0.4990)
    score = calibration.category_score(probs, CAL.post, CAL.post_margin)
    assert score.item() == pytest.approx(1 - _sig(0.7 * x), abs=1e-5)


def test_ratio_clipping() -> None:
    probs = _probs([0.0, 0.0, 1.0])
    assert calibration.binary_score(probs, CAL).item() == pytest.approx(
        _sig(0.8 * calibration.X_CLIP - 0.3), abs=1e-5
    )


def test_score_bands_match_geozarr_categories() -> None:
    """The GeoZarr category and the score branch agree on every pixel."""
    rng = np.random.default_rng(0)
    h = w = 16
    logits = {
        "binary": rng.normal(size=(3, h, w)),
        "pre": rng.normal(size=(len(PRE_CHANGE_CATEGORY_NAMES), h, w)) * 3,
        "post": rng.normal(size=(len(POST_CHANGE_CATEGORY_NAMES), h, w)) * 3,
    }
    probs = {
        k: torch.softmax(torch.tensor(v, dtype=torch.float32), 0)
        for k, v in logits.items()
    }
    scores = calibration.score_bands(probs["binary"], probs["pre"], probs["post"], CAL)
    arr = np.zeros((len(OUTPUT_BANDS), h, w), dtype=np.uint16)
    arr[0:3] = calibration.quantize(probs["binary"]).numpy()
    arr[
        PRE_CHANGE_BAND_OFFSET : PRE_CHANGE_BAND_OFFSET + len(PRE_CHANGE_CATEGORY_NAMES)
    ] = calibration.quantize(probs["pre"]).numpy()
    arr[
        POST_CHANGE_BAND_OFFSET : POST_CHANGE_BAND_OFFSET
        + len(POST_CHANGE_CATEGORY_NAMES)
    ] = calibration.quantize(probs["post"]).numpy()
    arr[SCORE_BAND_OFFSET : SCORE_BAND_OFFSET + 3] = calibration.quantize(
        scores
    ).numpy()
    change, _ = geozarr.geozarr_window_arrays(arr)
    post_cat = change[geozarr.CHANGE_BANDS.index("post_change_category")]
    p = probs["post"].numpy()
    x = np.log10(np.clip(np.sort(p[2:], 0)[-1], 6e-8, None) / np.clip(p[1], 6e-8, None))
    p_change = 1 / (1 + np.exp(-(0.7 * np.clip(x, -7.2, 7.2))))
    none = post_cat == NONE_CATEGORY_IDX
    kept = change[geozarr.CHANGE_BANDS.index("post_lc")] > 0
    np.testing.assert_allclose(scores[2].numpy()[none], 1 - p_change[none], atol=1e-5)
    assert (scores[2].numpy()[~none] <= p_change[~none] + 1e-6).all()
    # Stored post score = quantized score band where details are kept.
    np.testing.assert_array_equal(
        change[geozarr.CHANGE_BANDS.index("post_change_score")][kept],
        arr[SCORE_BAND_OFFSET + 2][kept],
    )


def test_read_score_calibration(tmp_path: pathlib.Path) -> None:
    cfg = {
        "data": {
            "init_args": {"task": {"init_args": {"score_calibration": CAL.to_dict()}}}
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    read = calibration.read_score_calibration(str(path))
    assert read is not None
    assert ScoreCalibration.from_dict(read) == CAL
    cfg["data"]["init_args"]["task"]["init_args"] = {}
    path.write_text(yaml.safe_dump(cfg))
    assert calibration.read_score_calibration(str(path)) is None


def test_fit_score_calibration_shift() -> None:
    """The deployment shift is -logit(pi_val) + log(FPR_val) - log(1 - R)."""
    rng = np.random.default_rng(1)
    n = 2000
    binary = rng.choice([1, 2], size=n)
    change = binary == 2
    z = np.where(change, rng.normal(2, 1.5, n), rng.normal(-3, 1.5, n))
    p_change = 1 / (1 + np.exp(-z))
    probs_binary = np.stack([np.zeros(n), 1 - p_change, p_change], 1)
    num_post = len(POST_CHANGE_CATEGORY_NAMES)
    num_pre = len(PRE_CHANGE_CATEGORY_NAMES)
    # Some labels disagree with the top category, so the margin fit has both outcomes.
    post_label = np.where(change, rng.choice([3, 4], size=n, p=[0.8, 0.2]), 0)
    probs_post = np.full((n, num_post), 1e-4)
    probs_post[:, NONE_CATEGORY_IDX] = 1 - p_change
    probs_post[:, 3] = p_change * 0.7
    probs_post[:, 4] = p_change * 0.3
    probs_pre = np.full((n, num_pre), 1e-4)
    probs_pre[:, NONE_CATEGORY_IDX] = 1 - p_change
    probs_pre[:, 2] = p_change * 0.9
    probs_pre[:, 3] = p_change * 0.1
    pre_label = np.where(change, rng.choice([2, 3], size=n, p=[0.9, 0.1]), 0)
    cal, stats = calibration.fit_score_calibration(
        probs_binary,
        probs_pre,
        probs_post,
        binary,
        pre_label,
        post_label,
        np.ones(n),
        np.ones(n),
        source="synthetic",
    )
    for head in ("binary", "pre", "post"):
        s = stats[head]
        expected = (
            -math.log(s["pi_val"] / (1 - s["pi_val"]))
            + math.log(s["fpr_val"])
            - math.log(1 - s["recall"])
        )
        assert s["shift"] == pytest.approx(expected)
        assert getattr(cal, head).b == pytest.approx(s["val_curve"]["b"] + expected)
        assert getattr(cal, head).a > 0
    assert cal.source == "synthetic"
