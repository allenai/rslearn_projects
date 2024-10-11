"""NMS for merging predictions from multiple patches."""

import math

import numpy as np
from rslearn.train.prediction_writer import PatchPredictionMerger
from rslearn.utils import Feature, GridIndex

# Defaults for distance-based NMS
DEFAULT_GRID_SIZE = 64
DEFAULT_DISTANCE_THRESHOLD = 10


class NMSDistanceMerger(PatchPredictionMerger):
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

    def merge(self, features: list[Feature]) -> list[Feature]:
        """Merge the predictions from multiple patches.

        Args:
            features: predictions (vector data) to merge.

        Returns:
            the merged vector data.
        """
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
        print(f"Kept {len(keep_indices)} out of {len(features)} features after NMS")

        return [features[i] for i in keep_indices]

    def _boxes_center_distance(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute the Euclidean distance between the centers of two boxes.

        Args:
            box1: numpy array of shape (4,) representing the first bounding box.
            box2: numpy array of shape (4,) representing the second bounding box.

        Returns:
            distance: the Euclidean distance between the centers of the two boxes.
        """
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        dx = cx1 - cx2
        dy = cy1 - cy2
        return math.sqrt(dx * dx + dy * dy)

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
        if indices is None:
            indices = np.arange(len(boxes))

        grid_index = GridIndex(size=max(self.grid_size, self.distance_threshold))
        for idx, box in zip(indices, boxes):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            grid_index.insert((cx, cy, cx, cy), idx)

        sorted_order = np.argsort(scores)
        sorted_indices = indices[sorted_order]
        sorted_boxes = boxes[sorted_order]
        sorted_scores = scores[sorted_order]

        elim_inds = set()
        keep_indices = []
        for idx, box, score in zip(sorted_indices, sorted_boxes, sorted_scores):
            if idx in elim_inds:
                continue
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            rect = [
                cx - self.distance_threshold,
                cy - self.distance_threshold,
                cx + self.distance_threshold,
                cy + self.distance_threshold,
            ]
            neighbor_indices = grid_index.query(rect)
            for other_idx in neighbor_indices:
                i = np.where(sorted_indices == other_idx)[0][0]
                if other_idx == idx or other_idx in elim_inds:
                    continue
                other_score = sorted_scores[i]
                if other_score > score or (other_score == score and other_idx < idx):
                    other_box = sorted_boxes[i]
                    distance = self._boxes_center_distance(box, other_box)
                    if distance <= self.distance_threshold:
                        elim_inds.add(idx)
                        break
            if idx not in elim_inds:
                keep_indices.append(idx)

        return keep_indices
