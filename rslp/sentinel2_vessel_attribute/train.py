"""Custom task and augmentation for vessel attribute trianing."""

import math
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import rslearn.main
import torch
import wandb
from PIL import Image, ImageDraw
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.task import BasicTask, Task
from rslearn.utils import Feature
from torchmetrics import Metric, MetricCollection

SHIP_TYPE_CATEGORIES = [
    "cargo",
    "tanker",
    "passenger",
    "service",
    "tug",
    "pleasure",
    "fishing",
    "enforcement",
    "sar",
]


class HeadingMetric(Metric):
    """Metric for heading which comes from heading_x and heading_y combination."""

    def __init__(self, degrees_tolerance: float = 10):
        """Create a new HeadingMetric.

        Args:
            degrees_tolerance: consider prediction correct as long as it is within this
                many degrees of the ground truth.
        """
        super().__init__()
        self.degrees_tolerance = degrees_tolerance
        self.correct = 0
        self.total = 0

    def update(
        self, preds: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        for output, target_dict in zip(preds, targets):
            if not target_dict["heading_x"]["valid"]:
                continue

            pred_cog = (
                math.atan2(output["heading_y"], output["heading_x"]) * 180 / math.pi
            )
            gt_cog = (
                math.atan2(
                    target_dict["heading_y"]["value"],
                    target_dict["heading_x"]["value"],
                )
                * 180
                / math.pi
            )

            angle_difference = abs(pred_cog - gt_cog) % 360
            if angle_difference > 180:
                angle_difference = 360 - angle_difference

            if angle_difference <= self.degrees_tolerance:
                self.correct += 1
            self.total += 1

    def compute(self) -> Any:
        """Returns the computed metric."""
        return torch.tensor(self.correct / self.total)

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self.correct = 0
        self.total = 0

    def plot(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Returns a plot of the metric."""
        return None


class VesselAttributeMultiTask(MultiTask):
    """Extension of MultiTask with custom input pre-processing and visualization."""

    def __init__(
        self,
        tasks: dict[str, Task],
        input_mapping: dict[str, dict[str, str]],
        length_buckets: list[float] = [],
        width_buckets: list[float] = [],
        speed_buckets: list[float] = [],
    ):
        """Create a new VesselAttributeMultiTask.

        Args:
            tasks: see MultiTask.
            input_mapping: see MultiTask.
            length_buckets: which buckets to use for length attribute.
            width_buckets: which buckets to use for width attribute.
            speed_buckets: which attributes to use for speed attribute.
        """
        super().__init__(tasks, input_mapping)
        self.buckets = dict(
            length=length_buckets,
            width=width_buckets,
            speed=speed_buckets,
        )

    def _get_bucket(self, buckets: list[float], value: float) -> int:
        """Get bucket that the value belongs to.

        Args:
            buckets: a list of the values that separate buckets. For example, [1, 5]
                means there are three buckets, #0 covering 0-1, #1 for 1-5, and #2 for
                5+.
            value: the value to bucketize.

        Returns:
            the bucket index that the value belongs to.
        """
        for bucket_idx, threshold in enumerate(buckets):
            if value <= threshold:
                return bucket_idx
        return len(buckets)

    def _get_bucket_range(self, buckets: list[float], bucket_idx: int) -> str:
        """Get string representation of the range of a bucket.

        Args:
            buckets: a list of the values that separate buckets.
            bucket_idx: the bucket index to get range of.

        Returns:
            string representation of the range of the bucket like "5-10" or "10+".
        """
        if bucket_idx == len(buckets):
            return f"{buckets[bucket_idx-1]}+"

        if bucket_idx == 0:
            lo = 0.0
            hi = buckets[bucket_idx]
        else:
            lo = buckets[bucket_idx - 1]
            hi = buckets[bucket_idx]
        return f"{lo}-{hi}"

    def process_inputs(
        self,
        raw_inputs: dict[str, torch.Tensor | list[Feature]],
        metadata: dict[str, Any],
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Processes the data into targets.

        Args:
            raw_inputs: raster or vector data to process
            metadata: metadata about the patch being read
            load_targets: whether to load the targets or only inputs

        Returns:
            tuple (input_dict, target_dict) containing the processed inputs and targets
                that are compatible with both metrics and loss functions
        """
        # Add cog x/y components and then pass to superclass.
        if load_targets:
            for feat in raw_inputs["info"]:
                if "cog" in feat.properties:
                    angle = 90 - feat.properties["cog"]
                    feat.properties["cog_x"] = math.cos(angle * math.pi / 180)
                    feat.properties["cog_y"] = math.sin(angle * math.pi / 180)

                for task in ["length", "width", "speed"]:
                    if task == "speed":
                        prop_name = "sog"
                    else:
                        prop_name = task

                    if prop_name not in feat.properties:
                        continue
                    feat.properties[f"{prop_name}_bucket"] = self._get_bucket(
                        self.buckets[task], feat.properties[prop_name]
                    )

        return super().process_inputs(raw_inputs, metadata, load_targets)

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: dict[str, Any],
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets.

        Args:
            input_dict: the input dict from process_inputs
            target_dict: the target dict from process_inputs
            output: the prediction

        Returns:
            a dictionary mapping image name to visualization image
        """
        # Create combined visualization showing all the attributes.
        basic_task = BasicTask(remap_values=[[0.0, 0.3], [0, 255]])
        scale_factor = 0.01

        image = basic_task.visualize(input_dict, target_dict, output)["image"]
        image = image.repeat(axis=0, repeats=8).repeat(axis=1, repeats=8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        assert target_dict

        # Focus on specific mis-predictions.
        # if not target_dict["ship_type"]["valid"]:
        #     return {}
        # if abs(target_dict["speed"]["class"] - output["speed"].argmax()) <= 2:
        #    return {}

        lines = []
        for task in ["length", "width", "speed"]:
            if output[task].shape == ():
                # regression
                s = f"{task}: {output[task]/scale_factor:.1f}"
                if target_dict[task]["valid"]:
                    s += f" ({target_dict[task]['value']/scale_factor:.1f})"

            else:
                # classification
                output_bucket_idx = output[task].argmax().item()
                output_bucket_range = self._get_bucket_range(
                    self.buckets[task], output_bucket_idx
                )
                s = f"{task}: {output_bucket_range}"
                if target_dict[task]["valid"]:
                    target_bucket_range = self._get_bucket_range(
                        self.buckets[task], target_dict[task]["class"]
                    )
                    s += f" ({target_bucket_range})"

            lines.append(s)

        for task in ["heading"]:
            pred_cog = (
                math.atan2(output[task + "_y"], output[task + "_x"]) * 180 / math.pi
            )
            s = f"{task}: {pred_cog:.1f}"
            if target_dict[task + "_x"]["valid"]:
                gt_cog = (
                    math.atan2(
                        target_dict[task + "_y"]["value"],
                        target_dict[task + "_x"]["value"],
                    )
                    * 180
                    / math.pi
                )
                s += f" ({gt_cog:.1f})"
            lines.append(s)

        # angle_difference = abs(pred_cog - gt_cog) % 360
        # if angle_difference > 180:
        #    angle_difference = 360 - angle_difference
        # if angle_difference < 20:
        #    return {}

        for task in ["ship_type"]:
            pred_category = SHIP_TYPE_CATEGORIES[output[task].argmax()]
            s = f"{task}: {pred_category}"
            if target_dict[task]["valid"]:
                gt_category = SHIP_TYPE_CATEGORIES[target_dict[task]["class"]]
                s += f" ({gt_category})"
            lines.append(s)

        # only visualize cargo/tanker <-> fishing mis-predictions
        # okay1 = pred_category == "fishing" and gt_category in ["cargo", "tanker"]
        # okay2 = pred_category in ["cargo", "tanker"] and gt_category == "fishing"
        # if not (okay1 or okay2):
        #    return {}

        text = "\n".join(lines)
        box = draw.textbbox(xy=(0, 0), text=text, font_size=12)
        draw.rectangle(xy=box, fill=(0, 0, 0))
        draw.text(xy=(0, 0), text=text, font_size=12, fill=(255, 255, 255))
        return {
            "image": np.array(image),
        }

    def get_metrics(self) -> MetricCollection:
        """Get metrics for this task."""
        metrics = super().get_metrics()
        metrics.add_metrics({"heading_accuracy": HeadingMetric()})
        return metrics


class VesselAttributeLightningModule(RslearnLightningModule):
    """Extend LM to produce confusion matrices for each attribute."""

    def on_validation_epoch_start(self) -> None:
        """Called when at beginning of validation epoch.

        Here we initialize the confusion matrix.
        """
        self.val_probs: list[npt.NDArray[npt.float32]] = []
        self.val_y_true: list[npt.NDArray[np.int32]] = []

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Validation step extended with confusion matrix."""
        # Code below is copied from RslearnLightningModule.validation_step.
        inputs, targets, _ = batch
        batch_size = len(inputs)
        outputs, loss_dict = self(inputs, targets)
        val_loss = sum(loss_dict.values())
        self.log_dict(
            {"val_" + k: v for k, v in loss_dict.items()},
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_loss",
            val_loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.val_metrics.update(outputs, targets)
        self.log_dict(self.val_metrics, batch_size=batch_size, on_epoch=True)

        # Now we hook in part to compute confusion matrix.
        for output, target in zip(outputs, targets):
            if not target["ship_type"]["valid"]:
                continue
            self.val_probs.append(output["ship_type"].cpu().numpy())
            self.val_y_true.append(target["ship_type"]["class"].cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        """Push the confusion matrix to W&B."""
        self.logger.experiment.log(
            {
                "val_type_cm": wandb.plot.confusion_matrix(
                    probs=np.stack(self.val_probs),
                    y_true=np.stack(self.val_y_true),
                    class_names=SHIP_TYPE_CATEGORIES,
                )
            }
        )

    def on_test_epoch_start(self) -> None:
        """Called when at beginning of test epoch.

        Here we initialize the confusion matrices.
        """
        self.test_type_probs: list[npt.NDArray[npt.float32]] = []
        self.test_type_y_true: list[npt.NDArray[np.int32]] = []

        self.test_length_probs: list[npt.NDArray[npt.float32]] = []
        self.test_length_gt: list[npt.NDArray[np.int32]] = []
        self.test_width_probs: list[npt.NDArray[npt.float32]] = []
        self.test_width_gt: list[npt.NDArray[np.int32]] = []
        self.test_speed_probs: list[npt.NDArray[npt.float32]] = []
        self.test_speed_gt: list[npt.NDArray[np.int32]] = []

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Test step extended with confusion matrix."""
        # Code below is copied from RslearnLightningModule.test_step.
        inputs, targets, metadatas = batch
        batch_size = len(inputs)
        outputs, loss_dict = self(inputs, targets)
        test_loss = sum(loss_dict.values())
        self.log_dict(
            {"test_" + k: v for k, v in loss_dict.items()},
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_loss", test_loss, batch_size=batch_size, on_step=False, on_epoch=True
        )
        self.test_metrics.update(outputs, targets)
        self.log_dict(self.test_metrics, batch_size=batch_size, on_epoch=True)

        if self.visualize_dir:
            for idx, (inp, target, output, metadata) in enumerate(
                zip(inputs, targets, outputs, metadatas)
            ):
                images = self.task.visualize(inp, target, output)
                for image_suffix, image in images.items():
                    out_fname = os.path.join(
                        self.visualize_dir,
                        f'{metadata["window_name"]}_{metadata["bounds"][0]}_{metadata["bounds"][1]}_{image_suffix}.png',
                    )
                    Image.fromarray(image).save(out_fname)

        # Now we hook in part to compute confusion matrix.
        for output, target in zip(outputs, targets):
            if not target["ship_type"]["valid"]:
                continue
            self.test_type_probs.append(output["ship_type"].cpu().numpy())
            self.test_type_y_true.append(target["ship_type"]["class"].cpu().numpy())

        # Create other confusion matrices too.
        if output["length"].shape != ():
            for output, target in zip(outputs, targets):
                if not target["length"]["valid"]:
                    continue
                self.test_length_probs.append(output["length"].cpu().numpy())
                self.test_length_gt.append(target["length"]["class"].cpu().numpy())
        if output["width"].shape != ():
            for output, target in zip(outputs, targets):
                if not target["width"]["valid"]:
                    continue
                self.test_width_probs.append(output["width"].cpu().numpy())
                self.test_width_gt.append(target["width"]["class"].cpu().numpy())
        if output["speed"].shape != ():
            for output, target in zip(outputs, targets):
                if not target["speed"]["valid"]:
                    continue
                self.test_speed_probs.append(output["speed"].cpu().numpy())
                self.test_speed_gt.append(target["speed"]["class"].cpu().numpy())

    def on_test_epoch_end(self) -> None:
        """Push the confusion matrices to W&B."""
        self.logger.experiment.log(
            {
                "test_type_cm": wandb.plot.confusion_matrix(
                    probs=np.stack(self.test_type_probs),
                    y_true=np.stack(self.test_type_y_true),
                    class_names=SHIP_TYPE_CATEGORIES,
                )
            }
        )
        if len(self.test_length_probs) > 0:
            num_buckets = max(self.test_length_gt) + 1
            self.logger.experiment.log(
                {
                    "test_length_cm": wandb.plot.confusion_matrix(
                        probs=np.stack(self.test_length_probs),
                        y_true=np.stack(self.test_length_gt),
                        class_names=[f"bucket{idx}" for idx in range(num_buckets)],
                    )
                }
            )
        if len(self.test_width_probs) > 0:
            num_buckets = max(self.test_width_gt) + 1
            self.logger.experiment.log(
                {
                    "test_width_cm": wandb.plot.confusion_matrix(
                        probs=np.stack(self.test_width_probs),
                        y_true=np.stack(self.test_width_gt),
                        class_names=[f"bucket{idx}" for idx in range(num_buckets)],
                    )
                }
            )
        if len(self.test_speed_probs) > 0:
            num_buckets = max(self.test_speed_gt) + 1
            self.logger.experiment.log(
                {
                    "test_speed_cm": wandb.plot.confusion_matrix(
                        probs=np.stack(self.test_speed_probs),
                        y_true=np.stack(self.test_speed_gt),
                        class_names=[f"bucket{idx}" for idx in range(num_buckets)],
                    )
                }
            )


class VesselAttributeFlip(torch.nn.Module):
    """Flip inputs horizontally and/or vertically.

    Also extracts x/y component from the heading.
    """

    def __init__(
        self,
        horizontal: bool = True,
        vertical: bool = True,
    ):
        """Initialize a new MyFlip.

        Args:
            horizontal: whether to randomly flip horizontally
            vertical: whether to randomly flip vertically
        """
        super().__init__()
        self.horizontal = horizontal
        self.vertical = vertical

    def sample_state(self) -> dict[str, bool]:
        """Randomly decide how to transform the input.

        Returns:
            dict of sampled choices
        """
        horizontal = False
        if self.horizontal:
            horizontal = torch.randint(low=0, high=2, size=()) == 0
        vertical = False
        if self.vertical:
            vertical = torch.randint(low=0, high=2, size=()) == 0
        return {
            "horizontal": horizontal,
            "vertical": vertical,
        }

    def apply_state(
        self,
        state: dict[str, bool],
        d: dict[str, Any],
        image_keys: list[str],
        heading_keys: list[str],
    ) -> None:
        """Apply the flipping.

        Args:
            state: the sampled state from sample_state.
            d: the input or target dict.
            image_keys: image keys to flip.
            heading_keys: heading keys to flip.
        """
        for k in image_keys:
            if state["horizontal"]:
                d[k] = torch.flip(d[k], dims=[-1])
            if state["vertical"]:
                d[k] = torch.flip(d[k], dims=[-2])

        for k in heading_keys:
            if state["horizontal"]:
                d[k + "_x"]["value"] *= -1
            if state["vertical"]:
                d[k + "_y"]["value"] *= -1

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dicts, target_dicts) tuple
        """
        state = self.sample_state()
        self.apply_state(state, input_dict, ["image"], [])
        self.apply_state(state, target_dict, [], ["heading"])
        return input_dict, target_dict


if __name__ == "__main__":
    rslearn.main.main()
