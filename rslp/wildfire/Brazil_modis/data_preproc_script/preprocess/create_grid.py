"""Create a geographic grid of potential samples over an arbitrary bounding box."""

import shutil
from pathlib import Path

import geopandas as gpd
import hydra
from omegaconf import DictConfig

from data_preproc_script.constants import CONFIG_PATH
from data_preproc_script.preprocess.utils import adaptive_grid, grid


def create_grid(
    grid_type: str,
    output_path: str | Path,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    step_x: float,
    step_y: float,
    crs: int = 4326,
) -> gpd.GeoDataFrame:
    """Compute a geographic grid and save it to disk.

    Supports two grid types:
        - ``"fixed"``: cells defined by longitude/latitude step sizes (degrees).
        - ``"adaptive"``: cells defined by step sizes in **meters**, so that
          every cell covers roughly the same ground area regardless of latitude.

    Args:
        grid_type: Type of grid to create. Must be ``"fixed"`` or ``"adaptive"``.
        output_path: Destination file path for the GeoJSON output.
        min_x: Minimum longitude of the bounding box.
        max_x: Maximum longitude of the bounding box.
        min_y: Minimum latitude of the bounding box.
        max_y: Maximum latitude of the bounding box.
        step_x: Grid cell width. In degrees for ``"fixed"``, in meters for
            ``"adaptive"``.
        step_y: Grid cell height. In degrees for ``"fixed"``, in meters for
            ``"adaptive"``.
        crs: EPSG code for the coordinate reference system. Defaults to 4326
            (WGS 84).

    Returns:
        The generated grid as a GeoDataFrame.

    Raises:
        ValueError: If *grid_type* is not ``"fixed"`` or ``"adaptive"``.
    """
    if grid_type == "fixed":
        result_grid = grid(min_x, max_x, min_y, max_y, step_x, step_y)
    elif grid_type == "adaptive":
        result_grid = adaptive_grid(
            min_x, max_x, min_y, max_y, int(step_x), int(step_y)
        )
    else:
        raise ValueError(
            f"Invalid grid type: {grid_type!r}. Must be 'fixed' or 'adaptive'."
        )

    result_grid = result_grid.set_crs(epsg=crs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.with_suffix(".gdb").exists():
        shutil.rmtree(output_path.with_suffix(".gdb"))
    result_grid.to_file(output_path.with_suffix(".gdb"))

    return result_grid


@hydra.main(
    version_base=None, config_path=str(CONFIG_PATH), config_name="global_config"
)
def create_grid_wrapper(cfg: DictConfig) -> None:
    """Entry point that reads grid parameters from global_config.yaml."""
    create_grid(
        grid_type=cfg.grid.type,
        output_path=cfg.grid.output_path,
        min_x=cfg.bounds.min_x,
        max_x=cfg.bounds.max_x,
        min_y=cfg.bounds.min_y,
        max_y=cfg.bounds.max_y,
        step_x=cfg.grid.step_x,
        step_y=cfg.grid.step_y,
        crs=cfg.grid.crs,
    )


if __name__ == "__main__":
    create_grid_wrapper()
