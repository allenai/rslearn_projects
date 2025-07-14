"""Custom task and augmentation for vessel attribute trianing."""

import math
import os
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import rslearn.main
import torch
import wandb
from PIL import Image, ImageDraw
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.regression import RegressionTask
from rslearn.train.tasks.task import BasicTask, Task
from rslearn.utils.feature import Feature
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


class HeadingMode(str, Enum):
    """The method by which we are representing the heading to the model.

    The heading here is represented as the angle counterclockwise from the positive x
    axis. It is derived from the north-based course over ground in
    VesselAttributeMultiTask.

    XY simply computes cos(angle) and sin(angle).

    XYD computes cos(2*angle) and sin(2*angle), which makes angle and (angle + 180)
    have the same x/y components. Then, it separately predicts P[angle > 180]. This
    way, the model does not have to guess as much about the direction of the vessel
    when predicting the x/y components.
    """

    XY = "xy"
    XYD = "xyd"

    def _get_xy_angle(self, x: float, y: float) -> float:
        """Returns the angle for XY mode given x/y components."""
        return math.atan2(y, x) * 180 / math.pi

    def _get_xyd_angle(self, x: float, y: float, direction: int) -> float:
        """Returns the angle for XYD mode given x/y/direction."""
        # Compute the angle that went into the cos/sin.
        angle = math.atan2(y, x) * 180 / math.pi
        # The original angle was doubled, so we need to halve the angle.
        # However the actual angle could be this one or the opposite.
        angle = angle / 2
        # Normalize it to be the smaller angle.
        angle = angle % 360
        if angle > 180:
            angle = angle - 180
        # Now if the direction is 1 then we need to flip it back.
        if direction == 1:
            angle = angle + 180
        return angle

    def get_output_angle(self, output: dict[str, Any]) -> float:
        """Get the predicted angle from the output dictionary."""
        if self == HeadingMode.XY:
            return self._get_xy_angle(
                output["heading_x"].item(),
                output["heading_y"].item(),
            )
        elif self == HeadingMode.XYD:
            return self._get_xyd_angle(
                output["heading2_x"].item(),
                output["heading2_y"].item(),
                output["heading2_direction"].argmax().item(),
            )
        # Should not be possible.
        assert False

    def get_target_angle(self, target: dict[str, Any]) -> float | None:
        """Get the angle label from the target dictionary.

        Returns None of the example did not have a label for the angle attribute.
        """
        if self == HeadingMode.XY:
            if not target["heading_x"]["valid"]:
                return None
            return self._get_xy_angle(
                target["heading_x"]["value"].item(),
                target["heading_y"]["value"].item(),
            )
        elif self == HeadingMode.XYD:
            if not target["heading2_x"]["valid"]:
                return None
            return self._get_xyd_angle(
                target["heading2_x"]["value"].item(),
                target["heading2_y"]["value"].item(),
                target["heading2_direction"]["class"].item(),
            )
        # Should not be possible.
        assert False


class HeadingMetric(Metric):
    """Metric for heading which comes from heading_x and heading_y combination."""

    def __init__(self, heading_mode: HeadingMode, degrees_tolerance: float = 10):
        """Create a new HeadingMetric.

        Args:
            heading_mode: how the heading is being represented.
            degrees_tolerance: consider prediction correct as long as it is within this
                many degrees of the ground truth.
        """
        super().__init__()
        self.heading_mode = heading_mode
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
            gt_cog = self.heading_mode.get_target_angle(target_dict)
            if gt_cog is None:
                # Means this task is invalid for this example.
                continue
            pred_cog = self.heading_mode.get_output_angle(output)

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
        heading_mode: HeadingMode = HeadingMode.XY,
    ):
        """Create a new VesselAttributeMultiTask.

        Args:
            tasks: see MultiTask.
            input_mapping: see MultiTask.
            length_buckets: which buckets to use for length attribute.
            width_buckets: which buckets to use for width attribute.
            speed_buckets: which attributes to use for speed attribute.
            heading_mode: how heading should be predicted
        """
        super().__init__(tasks, input_mapping)
        self.heading_mode = heading_mode
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
        # Add various derived properties to support different versions of tasks, like
        # predicting cog x/y components with regression (instead of directly predicting
        # the angle) and classifying the length/width/speed by bucket.
        # Then we pass to superclass to handle what each sub-task needs.
        if load_targets:
            for feat in raw_inputs["info"]:
                if "cog" in feat.properties:
                    angle = 90 - feat.properties["cog"]
                    if self.heading_mode == HeadingMode.XY:
                        # Compute x/y components of the angle.
                        feat.properties["cog_x"] = math.cos(angle * math.pi / 180)
                        feat.properties["cog_y"] = math.sin(angle * math.pi / 180)

                    if self.heading_mode == HeadingMode.XYD:
                        # Compute x/y/direction (see that HeadingMode for details).
                        feat.properties["cog2_x"] = math.cos(angle * math.pi / 180 * 2)
                        feat.properties["cog2_y"] = math.sin(angle * math.pi / 180 * 2)
                        if angle % 360 > 180:
                            feat.properties["cog2_direction"] = 1
                        else:
                            feat.properties["cog2_direction"] = 0

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

    def process_output(self, raw_output: Any, metadata: dict[str, Any]) -> Feature:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head.
            metadata: metadata about the patch being read

        Returns:
            either raster or vector data.
        """
        # Merge the Features from the regression and classification tasks into a single
        # feature that has all of those properties.
        feature = None
        for task_name, task in self.tasks.items():
            task_output = task.process_output(raw_output[task_name], metadata)
            task_feature = task_output[0]
            if not isinstance(task_feature, Feature):
                raise ValueError(
                    f"expected task {task_name} to output a Feature but got {task_feature}"
                )
            if feature is None:
                feature = task_feature
            else:
                feature.properties.update(task_feature.properties)

        # Add heading based on the heading mode.
        angle = self.heading_mode.get_output_angle(raw_output)
        assert feature is not None
        feature.properties["heading"] = 90 - angle

        return [feature]

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

        # only visualize heading mis-predictions
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
        metrics.add_metrics(
            {"heading_accuracy": HeadingMetric(heading_mode=self.heading_mode)}
        )
        return metrics


class VesselAttributeLightningModule(RslearnLightningModule):
    """Extend LM to produce confusion matrices for each attribute."""

    def on_test_epoch_start(self) -> None:
        """Called when at beginning of test epoch.

        Here we initialize the confusion matrices.
        """
        self.test_cm_probs: dict[str, list[npt.NDArray[npt.float32]]] = {
            "ship_type": [],
            "length": [],
            "width": [],
            "speed": [],
        }
        self.test_cm_gt: dict[str, list[npt.NDArray[np.int32]]] = {
            "ship_type": [],
            "length": [],
            "width": [],
            "speed": [],
        }

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

        # Now we hook in part to compute confusion matrices.
        # For length/width/speed, they could be either classification task or
        # regression. If it is regression, we need to convert the predicted and gt
        # values to a class (bucket). The buckets are specified in the Task.
        vessel_attribute_multi_task = self.task
        assert isinstance(vessel_attribute_multi_task, VesselAttributeMultiTask)

        for output, target in zip(outputs, targets):
            if target["ship_type"]["valid"]:
                self.test_cm_probs["ship_type"].append(
                    output["ship_type"].cpu().numpy()
                )
                self.test_cm_gt["ship_type"].append(
                    target["ship_type"]["class"].cpu().numpy()
                )

            for task_name in ["length", "width", "speed"]:
                if not target[task_name]["valid"]:
                    continue

                if output[task_name].shape == ():
                    # This means it is using regression (output is a scalar).
                    # So we need to convert to bucket.
                    buckets = vessel_attribute_multi_task.buckets[task_name]
                    sub_task = vessel_attribute_multi_task.tasks[task_name]
                    assert isinstance(sub_task, RegressionTask)
                    scale_factor = sub_task.scale_factor
                    output_bucket = vessel_attribute_multi_task._get_bucket(
                        buckets,
                        output[task_name].cpu().numpy() / scale_factor,
                    )
                    # Make fake probabilities for it.
                    output_probs = np.zeros((len(buckets) + 1,), dtype=np.float32)
                    output_probs[output_bucket] = 1

                    gt_bucket = vessel_attribute_multi_task._get_bucket(
                        buckets,
                        target[task_name]["value"].cpu().numpy() / scale_factor,
                    )
                    self.test_cm_probs[task_name].append(output_probs)
                    self.test_cm_gt[task_name].append(gt_bucket)

                else:
                    # It is classification so it is already in buckets.
                    self.test_cm_probs[task_name].append(
                        output[task_name].cpu().numpy()
                    )
                    self.test_cm_gt[task_name].append(
                        target[task_name]["class"].cpu().numpy()
                    )

    def on_test_epoch_end(self) -> None:
        """Push the confusion matrices to W&B."""
        vessel_attribute_multi_task = self.task
        assert isinstance(vessel_attribute_multi_task, VesselAttributeMultiTask)

        for task_name, probs_list in self.test_cm_probs.items():
            if len(probs_list) == 0:
                continue
            gt_list = self.test_cm_gt[task_name]

            if task_name == "ship_type":
                class_names = SHIP_TYPE_CATEGORIES
            else:
                buckets = vessel_attribute_multi_task.buckets[task_name]
                num_buckets = len(buckets) + 1
                class_names = [f"bucket{idx}" for idx in range(num_buckets)]

            self.logger.experiment.log(
                {
                    f"test_{task_name}_cm": wandb.plot.confusion_matrix(
                        probs=np.stack(probs_list),
                        y_true=np.stack(gt_list),
                        class_names=class_names,
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
        heading_mode: HeadingMode = HeadingMode.XY,
    ):
        """Initialize a new VesselAttributeFlip.

        Args:
            horizontal: whether to randomly flip horizontally
            vertical: whether to randomly flip vertically
            heading_mode: how the heading is being represented.
        """
        super().__init__()
        self.horizontal = horizontal
        self.vertical = vertical
        self.heading_mode = heading_mode

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
        update_heading: bool,
    ) -> None:
        """Apply the flipping.

        Args:
            state: the sampled state from sample_state.
            d: the input or target dict.
            image_keys: image keys to flip.
            update_heading: whether this is target and so heading should be updated.
        """
        for k in image_keys:
            if state["horizontal"]:
                d[k] = torch.flip(d[k], dims=[-1])
            if state["vertical"]:
                d[k] = torch.flip(d[k], dims=[-2])

        if update_heading:
            if self.heading_mode == HeadingMode.XY:
                if state["horizontal"]:
                    d["heading_x"]["value"] *= -1
                if state["vertical"]:
                    d["heading_y"]["value"] *= -1
            elif self.heading_mode == HeadingMode.XYD:
                # This case is more complicated so we just flip the original angle.
                angle = self.heading_mode.get_target_angle(d)

                if angle is not None:
                    if state["horizontal"]:
                        angle = 180 - angle
                    if state["vertical"]:
                        angle = -angle

                    d["heading2_x"]["value"].fill_(math.cos(angle * math.pi / 180 * 2))
                    d["heading2_y"]["value"].fill_(math.sin(angle * math.pi / 180 * 2))
                    if angle % 360 > 180:
                        d["heading2_direction"]["class"].fill_(1)
                    else:
                        d["heading2_direction"]["class"].fill_(0)
            else:
                raise ValueError(
                    f"unsupported flip for heading mode {self.heading_mode}"
                )

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
        self.apply_state(state, input_dict, ["image"], False)
        self.apply_state(state, target_dict, [], True)
        return input_dict, target_dict


if __name__ == "__main__":
    rslearn.main.main()
