"""NMS for merging predictions from multiple patches."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from rslearn.config import LayerConfig
from rslearn.dataset import Window
from rslearn.train.prediction_writer import CropPredictionMerger, PendingCropOutput
from rslearn.utils import GridIndex

# Defaults for distance-based NMS
DEFAULT_GRID_SIZE = 64
DEFAULT_DISTANCE_THRESHOLD = 10


def distance_nms(
    centers: np.ndarray,
    scores: np.ndarray,
    distance_threshold: int,
    grid_size: int = DEFAULT_GRID_SIZE,
    indices: np.ndarray | None = None,
) -> list[int]:
    """Apply distance-based non-maximum suppression over detection centers.

    Greedily keeps detections in descending score order, eliminating any detection whose
    center lies within ``distance_threshold`` of an already-kept, higher-scoring one.

    Args:
        centers: (N, 2) array of (x, y) detection centers.
        scores: (N,) array of detection scores.
        distance_threshold: detections within this center distance are considered the
            same object.
        grid_size: cell size for the spatial index used to look up nearby centers.
        indices: original indices for the rows of ``centers``, defaulting to range(N).
            Returned indices are drawn from this set.

    Returns:
        the indices of the detections to keep.
    """
    if indices is None:
        indices = np.arange(len(centers))

    grid_index = GridIndex(size=max(grid_size, distance_threshold))
    for idx, (cx, cy) in zip(indices, centers):
        grid_index.insert((cx, cy, cx, cy), idx)

    sorted_order = np.argsort(scores)
    sorted_indices = indices[sorted_order]
    sorted_centers = centers[sorted_order]
    sorted_scores = scores[sorted_order]

    elim_inds: set[int] = set()
    keep_indices: list[int] = []
    for idx, (cx, cy), score in zip(sorted_indices, sorted_centers, sorted_scores):
        if idx in elim_inds:
            continue
        rect = [
            cx - distance_threshold,
            cy - distance_threshold,
            cx + distance_threshold,
            cy + distance_threshold,
        ]
        for other_idx in grid_index.query(rect):
            i = np.where(sorted_indices == other_idx)[0][0]
            if other_idx == idx or other_idx in elim_inds:
                continue
            other_score = sorted_scores[i]
            if other_score > score or (other_score == score and other_idx < idx):
                other_cx, other_cy = sorted_centers[i]
                if math.hypot(cx - other_cx, cy - other_cy) <= distance_threshold:
                    elim_inds.add(idx)
                    break
        if idx not in elim_inds:
            keep_indices.append(idx)

    return keep_indices


class NMSDistanceMerger(CropPredictionMerger):
    """Merge predictions by applying distance-based NMS."""

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        distance_threshold: int = DEFAULT_DISTANCE_THRESHOLD,
        class_agnostic: bool = False,
        property_name: str = "category",
    ):
        """Create a new NMSDistanceMerger.

        Args:
            grid_size: size of the grid for distance NMS.
            distance_threshold: distance threshold for NMS.
            class_agnostic: whether to apply class-agnostic NMS.
            property_name: name of the property to apply NMS to.
        """
        self.grid_size = grid_size
        self.distance_threshold = distance_threshold
        self.class_agnostic = class_agnostic
        self.property_name = property_name

    def merge(
        self,
        window: Window,
        outputs: Sequence[PendingCropOutput],
        layer_config: LayerConfig,
    ) -> Any:
        """Merge the outputs.

        Args:
            window: the window we are merging the outputs for.
            outputs: the outputs to process.
            layer_config: the layer configuration.

        Returns:
            the merged outputs.
        """
        features = [feat for output in outputs for feat in output.output]
        if len(features) == 0:
            return []
        # TODO: load categories from config
        boxes = np.array([f.geometry.shp.bounds for f in features])
        scores = np.array([f.properties["score"] for f in features])
        class_ids = np.array([f.properties[self.property_name] for f in features])

        if self.class_agnostic:
            # Class-agnostic NMS: process all boxes together
            keep_indices = self._apply_nms(boxes, scores)
        else:
            keep_indices = []
            # Class-specific NMS: process boxes per class
            for class_id in np.unique(class_ids):
                idxs = np.where(class_ids == class_id)[0]
                if len(idxs) == 0:
                    continue
                class_boxes = boxes[idxs]
                class_scores = scores[idxs]
                class_keep_indices = self._apply_nms(class_boxes, class_scores, idxs)
                keep_indices.extend(class_keep_indices)
        # print how many are keeped out of total
        print(f"Kept {len(keep_indices)} out of {len(features)} detections after NMS")

        return [features[i] for i in keep_indices]

    def _apply_nms(
        self, boxes: np.ndarray, scores: np.ndarray, indices: np.ndarray = None
    ) -> list[int]:
        """Apply distance-based NMS to the given boxes and scores.

        Args:
            boxes: Array of bounding boxes.
            scores: Array of scores corresponding to the boxes.
            indices: Original indices of the boxes (optional).

        Returns:
            List of indices of boxes to keep.
        """
        centers = np.stack(
            [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1
        )
        return distance_nms(
            centers, scores, self.distance_threshold, self.grid_size, indices
        )
