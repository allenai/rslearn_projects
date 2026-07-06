"""Sentinel-1 vessel detection custom training code."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from rslearn.train.model_context import RasterImage
from rslearn.train.tasks.detection import DetectionTask


class Sentinel1VesselDetectionTask(DetectionTask):
    """Sentinel-1 vessel detection task.

    This behaves the same as the standard DetectionTask, except that the
    visualization renders every timestep of the input image (the current
    Sentinel-1 image plus any historical images concatenated along the time
    dimension) rather than only the first timestep. The same ground truth and
    predicted boxes are drawn on each timestep so the detections can be compared
    against the historical imagery.
    """

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets across all timesteps.

        Args:
            input_dict: the input dict from process_inputs
            target_dict: the target dict from process_inputs
            output: the prediction

        Returns:
            a dictionary mapping image name to visualization image
        """
        if target_dict is None:
            raise ValueError("target_dict is required for visualization")

        raster_image = input_dict["image"]
        assert isinstance(raster_image, RasterImage)
        num_timesteps = raster_image.image.shape[1]

        def get_image(timestep: int) -> npt.NDArray[Any]:
            image = raster_image.image.cpu()[self.image_bands, timestep, :, :]
            if self.remap_values:
                factor = (self.remap_values[1][1] - self.remap_values[1][0]) / (
                    self.remap_values[0][1] - self.remap_values[0][0]
                )
                image = (image - self.remap_values[0][0]) * factor + self.remap_values[
                    1
                ][0]
            return torch.clip(image, 0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)

        def draw_boxes(
            image: npt.NDArray[Any], d: dict[str, torch.Tensor]
        ) -> npt.NDArray[Any]:
            boxes = d["boxes"].cpu().numpy()
            class_ids = d["labels"].cpu().numpy()
            if "scores" in d:
                wanted = d["scores"].cpu().numpy() > self.score_threshold
                boxes = boxes[wanted]
                class_ids = class_ids[wanted]

            for box, class_id in zip(boxes, class_ids):
                sx = int(np.clip(box[0], 0, image.shape[1]))
                sy = int(np.clip(box[1], 0, image.shape[0]))
                ex = int(np.clip(box[2], 0, image.shape[1]))
                ey = int(np.clip(box[3], 0, image.shape[0]))
                color = self.colors[class_id % len(self.colors)]
                image[sy:ey, sx : sx + 2, :] = color
                image[sy:ey, ex - 2 : ex, :] = color
                image[sy : sy + 2, sx:ex, :] = color
                image[ey - 2 : ey, sx:ex, :] = color

            return image

        result: dict[str, npt.NDArray[Any]] = {}
        for timestep in range(num_timesteps):
            suffix = "" if timestep == 0 else f"_t{timestep}"
            base_image = get_image(timestep)
            result["gt" + suffix] = draw_boxes(base_image.copy(), target_dict)
            result["pred" + suffix] = draw_boxes(base_image.copy(), output)

        return result
