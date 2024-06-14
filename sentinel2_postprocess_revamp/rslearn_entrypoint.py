"""
Adds a custom task so that we get nicer visualizations.
Also Flip augmentation that doesn't mess up the heading.
"""
from typing import Any, Optional, Union

import math
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

import torch

import rslearn.main
from rslearn.train.tasks.task import BasicTask
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.utils import Feature

class MyMultiTask(MultiTask):
    def process_inputs(
        self, raw_inputs: dict[str, Union[npt.NDArray[Any], list[Feature]]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Add cog x/y components and then pass to superclass.
        for feat in raw_inputs["info"]:
            if "cog" not in feat.properties:
                continue
            angle = 90 - feat.properties["cog"]
            feat.properties["cog_x"] = math.cos(angle * math.pi / 180)
            feat.properties["cog_y"] = math.sin(angle * math.pi / 180)
        return super().process_inputs(raw_inputs)

    def visualize(
        self, input_dict: dict[str, Any], target_dict: Optional[dict[str, Any]], output: dict[str, Any]
    ) -> dict[str, npt.NDArray[Any]]:
        # Create combined visualization showing all the attributes.
        basic_task = BasicTask(remap_values=[[0.2, 0.5], [0, 255]])
        scale_factor = 0.01

        image = basic_task.visualize(input_dict, target_dict, output)["image"]
        image = image.repeat(axis=0, repeats=8).repeat(axis=1, repeats=8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        lines = []
        for task in ["length", "width", "speed"]:
            s = f"{task}: {output[task]/scale_factor:.1f}"
            if target_dict[task]["valid"]:
                s += f" ({target_dict[task]['value']/scale_factor:.1f})"
            lines.append(s)

        for task in ["heading"]:
            pred_cog = math.atan2(output[task + "_y"], output[task + "_x"]) * 180 / math.pi
            s = f"{task}: {pred_cog:.1f}"
            if target_dict[task + "_x"]["valid"]:
                gt_cog = math.atan2(target_dict[task + "_y"]["value"], target_dict[task + "_x"]["value"]) * 180 / math.pi
                s += f" ({gt_cog:.1f})"
            lines.append(s)

        categories = ["cargo", "tanker", "passenger", "service", "pleasure", "fishing", "enforcement", "sar"]
        for task in ["ship_type"]:
            s = f"{task}: {categories[output[task].argmax()]}"
            if target_dict[task]["valid"]:
                s += f" ({categories[target_dict[task]['class']]})"
            lines.append(s)

        text = "\n".join(lines)
        box = draw.textbbox(xy=(0, 0), text=text, font_size=12)
        draw.rectangle(xy=box, fill=(0, 0, 0))
        draw.text(xy=(0, 0), text=text, font_size=12, fill=(255, 255, 255))
        return {
            "image": np.array(image),
        }

class MyFlip(torch.nn.Module):
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
        self.generator = torch.Generator()

    def sample_state(self) -> dict[str, bool]:
        """Randomly decide how to transform the input.

        Returns:
            dict of sampled choices
        """
        horizontal = False
        if self.horizontal:
            horizontal = torch.randint(low=0, high=2, generator=self.generator, size=()) == 0
        vertical = False
        if self.vertical:
            vertical = torch.randint(low=0, high=2, generator=self.generator, size=()) == 0
        return {
            "horizontal": horizontal,
            "vertical": vertical,
        }

    def apply_state(self, state: dict[str, bool], d: dict[str, Any], image_keys: list[str], heading_keys: list[str]):
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

    def forward(self, input_dict, target_dict):
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
