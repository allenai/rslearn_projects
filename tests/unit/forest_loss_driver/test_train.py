from typing import Any

import pytest
import shapely
import torch
from rasterio.crs import CRS
from rslearn.models.component import FeatureVector
from rslearn.train.model_context import ModelContext, SampleMetadata
from rslearn.utils import Feature, Projection, STGeometry

from rslp.forest_loss_driver import hierarchy
from rslp.forest_loss_driver.train import (
    HierarchicalAccuracy,
    HierarchicalClassificationHead,
    HierarchicalForestLossTask,
)

PROJECTION = Projection(CRS.from_epsg(32718), 10, -10)
METADATA = SampleMetadata(
    window_group="group",
    window_name="window",
    window_bounds=(0, 0, 128, 128),
    crop_bounds=(0, 0, 128, 128),
    crop_idx=0,
    num_crops_in_window=1,
    time_range=None,
    projection=PROJECTION,
    dataset_source=None,
)


def _leaf_idx(leaf: str) -> int:
    return hierarchy.LEAVES.index(leaf)


def _target(leaves: list[str], new_label: str | None = None) -> dict[str, Any]:
    task = HierarchicalForestLossTask()
    properties: dict[str, Any] = {"leaf_labels": leaves}
    if new_label is not None:
        properties["new_label"] = new_label
    feature = Feature(STGeometry(PROJECTION, shapely.Point(0, 0), None), properties)
    _, target = task.process_inputs({"targets": [feature]}, METADATA)
    return target


def _head_loss(logits: torch.Tensor, targets: list[dict[str, Any]]) -> torch.Tensor:
    head = HierarchicalClassificationHead()
    context = ModelContext(
        inputs=[{}] * len(targets), metadatas=[METADATA] * len(targets)
    )
    output = head(FeatureVector(feature_vector=logits), context, targets)
    return output.loss_dict["cls"]


def test_process_inputs() -> None:
    target = _target(["road", "logging_road"], new_label="road")
    assert target["class"] == -1
    assert target["allowed"].sum() == 2
    assert target["allowed"][_leaf_idx("logging_road")] == 1
    assert target["legacy_class"] == hierarchy.LEGACY_CLASSES.index("road")
    assert target["valid"] == 1

    target = _target(["landslide"])
    assert target["class"] == _leaf_idx("landslide")
    assert target["legacy_class"] == -1

    # A label that allows every leaf carries no information.
    task = HierarchicalForestLossTask(allow_invalid=True)
    feature = Feature(
        STGeometry(PROJECTION, shapely.Point(0, 0), None),
        {"leaf_labels": list(hierarchy.LEAVES)},
    )
    _, target = task.process_inputs({"targets": [feature]}, METADATA)
    assert target["valid"] == 0
    assert target["allowed"].sum() == 0

    with pytest.raises(ValueError):
        HierarchicalForestLossTask().process_inputs({"targets": [feature]}, METADATA)


def test_loss_matches_cross_entropy_for_single_leaf() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, len(hierarchy.LEAVES))
    leaves = ["landslide", "agriculture_soybean", "no_change"]
    loss = _head_loss(logits, [_target([leaf]) for leaf in leaves])
    expected = torch.nn.functional.cross_entropy(
        logits, torch.tensor([_leaf_idx(leaf) for leaf in leaves])
    )
    assert torch.allclose(loss, expected)


def test_loss_sums_allowed_probabilities() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, len(hierarchy.LEAVES), requires_grad=True)
    leaves = ["road", "logging_road"]
    loss = _head_loss(logits, [_target(leaves)])
    probs = torch.softmax(logits, dim=1)[0]
    expected = -torch.log(probs[[_leaf_idx(leaf) for leaf in leaves]].sum())
    assert torch.allclose(loss, expected)

    # The loss does not care how the probability is split among the allowed leaves.
    loss.backward()
    assert logits.grad is not None
    grad = logits.grad[0]
    road_prob = probs[_leaf_idx("road")]
    logging_road_prob = probs[_leaf_idx("logging_road")]
    assert torch.allclose(
        grad[_leaf_idx("road")] / road_prob,
        grad[_leaf_idx("logging_road")] / logging_road_prob,
    )


def test_loss_ignores_invalid_examples() -> None:
    invalid = {
        "class": torch.tensor(-1),
        "allowed": torch.zeros(len(hierarchy.LEAVES)),
        "legacy_class": torch.tensor(-1),
        "valid": torch.tensor(0.0),
    }
    logits = torch.randn(2, len(hierarchy.LEAVES))
    loss = _head_loss(logits, [_target(["landslide"]), invalid])
    expected = torch.nn.functional.cross_entropy(
        logits[0:1], torch.tensor([_leaf_idx("landslide")])
    )
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected / 2)


def _one_hot_pred(leaf: str) -> torch.Tensor:
    pred = torch.zeros(len(hierarchy.LEAVES))
    pred[_leaf_idx(leaf)] = 1
    return pred


def test_accuracy_modes() -> None:
    targets = [
        _target(["road", "logging_road"], new_label="road"),
        _target(["landslide"], new_label="landslide"),
        _target(list(hierarchy.AGRICULTURE), new_label="agriculture"),
    ]
    preds = torch.stack(
        [
            _one_hot_pred("logging_road"),
            _one_hot_pred("blowdown"),
            _one_hot_pred("agriculture_rice"),
        ]
    )

    def compute(mode: str) -> float:
        metric = HierarchicalAccuracy(mode)
        metric.update(preds, targets)
        return metric.compute().item()

    assert compute("set") == pytest.approx(2 / 3)
    # Only the landslide example has a single leaf.
    assert compute("leaf") == 0
    # The road/logging road example spans two Level 2 nodes so it is excluded at that
    # level, but blowdown is still natural.
    assert compute("level1") == 1
    assert compute("level2") == pytest.approx(1 / 2)
    assert compute("legacy") == pytest.approx(2 / 3)


def test_process_output() -> None:
    task = HierarchicalForestLossTask(prob_property="probs")
    probs = torch.zeros(len(hierarchy.LEAVES))
    # The top leaf is landslide, but human-made has more probability overall.
    probs[_leaf_idx("landslide")] = 0.4
    probs[_leaf_idx("road")] = 0.3
    probs[_leaf_idx("logging_road")] = 0.3
    features = task.process_output(probs, METADATA)
    assert len(features) == 1
    properties = features[0].properties
    assert properties["new_label"] == "landslide"
    assert properties["level1"] == "human_made"
    assert properties["level2"] == "landslide"
    assert properties["legacy_category"] == "road"
    assert len(properties["probs"]) == len(hierarchy.LEAVES)
