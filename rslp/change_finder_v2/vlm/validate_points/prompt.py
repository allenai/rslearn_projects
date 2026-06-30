"""Prompt construction and image labeling for point validation.

The validation task asks Gemini whether a flagged location underwent a genuine,
long-term land-cover change at the center of the imagery, as opposed to seasonal or
phenological variation. We pass a short time series of images (one Sentinel-2 image and
up to one high-resolution aerial image per year), each captioned with its capture date.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

# Allowed values for the structured output.
ALLOWED_PREDICTIONS = ["positive", "negative"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]

_LABEL_STRIP_HEIGHT = 22
_MIN_DISPLAY_WIDTH = 256


@dataclass
class ImageRef:
    """A labeled image to include in the prompt, in order."""

    label: str
    png_bytes: bytes


def _get_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a legible font, falling back to PIL's default if needed."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
    except OSError:
        return ImageFont.load_default()


def label_image(chw_array: npt.NDArray, caption: str) -> bytes:
    """Render a CHW uint8 RGB array to PNG with a caption strip beneath it.

    Small images (e.g. 64x64 Sentinel-2 chips) are upscaled with nearest-neighbor so
    the caption text stays readable relative to the image.

    Args:
        chw_array: the image as a (C, H, W) array; the first three bands are used.
        caption: text drawn under the image (e.g. the image type and capture date).

    Returns:
        PNG-encoded bytes of the captioned image.
    """
    hwc = np.transpose(np.asarray(chw_array), (1, 2, 0))
    hwc = np.clip(hwc[:, :, :3], 0, 255).astype(np.uint8)
    image = Image.fromarray(hwc, "RGB")

    if image.width < _MIN_DISPLAY_WIDTH:
        scale = max(1, _MIN_DISPLAY_WIDTH // image.width)
        image = image.resize(
            (image.width * scale, image.height * scale), Image.NEAREST
        )

    out = Image.new("RGB", (image.width, image.height + _LABEL_STRIP_HEIGHT), (0, 0, 0))
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((4, image.height + 4), caption, fill=(255, 255, 255), font=_get_font())

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def build_validation_prompt(
    predicted_change_date: str | None,
    pre_category: str | None,
    post_category: str | None,
) -> str:
    """Build the text prompt for validating one flagged point.

    Args:
        predicted_change_date: the date the change was predicted/annotated to occur.
        pre_category: the land-cover category before the change, if known.
        post_category: the land-cover category after the change, if known.

    Returns:
        the prompt text.
    """
    lines = [
        "You are validating whether a real, long-term land-cover change occurred at a "
        "specific location, or whether the apparent difference is only seasonal or "
        "phenological variation.",
        "",
        "You are given a short time series of images of the SAME small area (about "
        "640 m across), all centered on the exact point of interest. For each year "
        "there is one Sentinel-2 satellite image (10 m resolution) and, when "
        "available, one high-resolution aerial image. Each image is captioned with its "
        "capture date. The images are provided in chronological order.",
        "",
        "A change-detection model flagged this point as having a long-term change.",
    ]

    details = []
    if predicted_change_date:
        details.append(f"the change was estimated to occur around {predicted_change_date}")
    if pre_category and post_category:
        details.append(
            f"the predicted transition is from '{pre_category}' to '{post_category}'"
        )
    if details:
        lines.append("For reference, " + "; ".join(details) + ".")

    lines += [
        "",
        "Focus only on the CENTER of each image. Decide:",
        "- 'positive' if there is a genuine, persistent land-cover change at the "
        "center: the post-change state appears and then remains in later images, and "
        "is not explained by season (e.g. crop growth/harvest, leaf-on/leaf-off, snow "
        "cover, water level, or temporary flooding).",
        "- 'negative' if the differences are explained by seasonal/phenological "
        "variation, clouds, haze, or image artifacts, or if there is no meaningful "
        "lasting change at the center.",
        "",
        "Give brief reasoning grounded in what the images show across the years, then "
        "your prediction and confidence.",
    ]
    return "\n".join(lines)
