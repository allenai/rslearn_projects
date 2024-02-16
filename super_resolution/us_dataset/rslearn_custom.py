from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt
from PIL import Image

import rslearn.main
from rslearn.utils import Projection, raster_format
from rslearn.utils.raster_format import RasterFormat


class CustomRasterFormat(RasterFormat):
    def encode_raster(
        self, f: BinaryIO, projection: Projection, tile: tuple[int, int], image: npt.NDArray[Any]
    ) -> None:
        image = image.transpose(1, 2, 0)[:, :, 0:3]
        Image.fromarray(image).save(f, format="PNG")


    def decode_raster(self, f: BinaryIO) -> npt.NDArray[Any]:
        """Decodes a raster tile."""
        return np.array(Image.open(f, format="PNG"))

    def get_extension(self):
        return "png"

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "CustomRasterFormat":
        return CustomRasterFormat()


raster_format.registry["custom"] = CustomRasterFormat


if __name__ == "__main__":
    rslearn.main.main()
