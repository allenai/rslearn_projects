"""Download MapBiomas Amazon land-cover classification maps for each year."""

import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import hydra
from omegaconf import DictConfig

from data_preproc_script.constants import CONFIG_PATH
from data_preproc_script.utils import create_logger

logger = create_logger("download_mapbiomas", "logs/download_mapbiomas")

URL_TEMPLATE = (
    "https://storage.googleapis.com/mapbiomas-public/initiatives/amazon/"
    "lulc/collection_6/integration/"
    "mapbiomas_collection60_integration_v1-classification_{year}.tif"
)


@hydra.main(
    version_base=None, config_path=str(CONFIG_PATH), config_name="ba_preprocess"
)
def main(cfg: DictConfig) -> None:
    """Download yearly MapBiomas land-cover rasters to the configured folder."""
    output_dir = Path(cfg.paths.landcover)
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in range(cfg.start_year, cfg.end_year + 1):
        url = URL_TEMPLATE.format(year=year)
        dest = output_dir / f"mapbiomas_classification_{year}.tif"

        if dest.exists():
            logger.info(f"Skipping {year} — already downloaded: {dest}")
            continue

        logger.info(f"Downloading {year}: {url}")
        print(f"[{year}] Downloading to {dest} ...")
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError(f"Unsupported download scheme for MapBiomas URL: {url}")
        urllib.request.urlretrieve(url, dest)  # nosec B310
        logger.info(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
