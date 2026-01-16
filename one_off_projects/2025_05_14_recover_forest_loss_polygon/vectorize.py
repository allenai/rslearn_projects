import multiprocessing
import numpy as np
import rasterio.features
import shapely
import tqdm
from PIL import Image
from rslearn.dataset import Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.vector_format import GeojsonVectorFormat, GeojsonCoordinateMode
from upath import UPath


def process_window(window_dir: UPath) -> None:
    """Vectorize mask.png and write it to the label data.geojson, for the given window.

    Args:
        window_dir: the window directory to process.
    """
    with (window_dir / "mask.png").open("rb") as f:
        mask = np.array(Image.open(f))

    if mask.max() == 0:
        # Some masks are empty because of some bug during data generation, in this case
        # we just stick to using a point.
        return

    # Vectorize using rasterio.features.shapes.
    shp: shapely.Geometry | None = None
    for cur_geo, cur_val in rasterio.features.shapes(mask):
        if cur_val == 0:
            continue
        cur_shp = shapely.geometry.shape(cur_geo)
        if shp is None or cur_shp.area > shp.area:
            shp = cur_shp

    if shp is None:
        raise ValueError(f"found no mask shape for window at {window_dir}")

    # Compute the polygon in global pixel coordinates by adding the window bounds.
    window = Window.load(window_dir)
    def add_window_offset(points: np.ndarray) -> np.ndarray:
        return points + np.array([window.bounds[0], window.bounds[1]])
    shp = shapely.transform(shp, add_window_offset)
    geom = STGeometry(window.projection, shp, None)

    # Add to the label data.geojson.
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    layer_name = "label"
    layer_dir = window.get_layer_dir(layer_name)
    features: list[Feature] = vector_format.decode_vector(layer_dir, window.projection, window.bounds)
    if len(features) != 1:
        raise ValueError(f"expected one feature in {layer_dir} but got {len(features)}")
    features[0].geometry = geom
    vector_format.encode_vector(layer_dir, features)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    ds_path = UPath("gs://rslearn-eai/datasets/forest_loss_driver/dataset_v1/20250514/")
    window_dirs = list(ds_path.glob("windows/*/*"))
    p = multiprocessing.Pool(128)
    outputs = p.imap_unordered(process_window, window_dirs)
    for _ in tqdm.tqdm(outputs, total=len(window_dirs)):
        pass
    p.close()
