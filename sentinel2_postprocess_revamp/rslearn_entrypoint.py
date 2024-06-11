"""
Adds a custom task so that we get nicer visualizations.
"""
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

import rslearn.main
from rslearn.train.tasks.task import BasicTask
from rslearn.train.tasks.multi_task import MultiTask

class MyMultiTask(MultiTask):
    def visualize(
        self, input_dict: dict[str, Any], target_dict: Optional[dict[str, Any]], output: dict[str, Any]
    ) -> dict[str, npt.NDArray[Any]]:
        basic_task = BasicTask(remap_values=[[0.2, 0.5], [0, 255]])
        scale_factor = 0.01

        image = basic_task.visualize(input_dict, target_dict, output)["image"]
        image = image.repeat(axis=0, repeats=8).repeat(axis=1, repeats=8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        lines = []
        for task in ["length", "width", "speed", "heading"]:
            s = f"{task}: {output[task]/scale_factor:.1f}"
            if target_dict[task]["valid"]:
                s += f" ({target_dict[task]['value']/scale_factor:.1f})"
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

if __name__ == "__main__":
    rslearn.main.main()
