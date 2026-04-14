"""Download FWI data from CEMS."""

from pathlib import Path

import cdsapi
import hydra
from omegaconf import DictConfig, OmegaConf

from data_preproc_script.constants import (
    CONFIG_PATH,
)
from data_preproc_script.utils import create_logger, fwi_agg

logger = create_logger("cds_data", "logs/cds_data")


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="fwi")
def cds_wrapper(cfg: DictConfig) -> None:
    """Wrapper function to download Copernicus Data."""
    logger.info("Downloading Copernicus data")
    c = cdsapi.Client()

    for year in range(cfg.start_year, cfg.end_year + 1):
        output_dir = Path(cfg.output_path) / str(year)
        output_dir.mkdir(exist_ok=True, parents=True)

        raw_file = output_dir / f"{cfg.product_name}-{year}.nc"
        agg_file = output_dir / f"fwi_dc_agg_{year}.nc"

        # Skip download if raw file already exists
        if raw_file.exists():
            logger.info(f"Raw file already exists, skipping download: {raw_file}")
        else:
            logger.info(
                f"Downloading CDS collection {cfg.product_name} for year {year}"
            )
            try:
                c.retrieve(
                    cfg.product_name,
                    {
                        "data_format": cfg.product_format,
                        "product_type": cfg.product_type,
                        "variable": OmegaConf.to_container(cfg.variable, resolve=True),
                        "system_version": [cfg.version],
                        "dataset_type": cfg.dataset,
                        "grid": cfg.grid,
                        "year": [str(year)],
                        "month": [f"{m:02d}" for m in range(1, 13)],
                        "day": [f"{d:02d}" for d in range(1, 32)],
                    },
                    str(raw_file),
                )
                logger.info(f"Successfully downloaded {raw_file}")
            except Exception as e:
                logger.error(
                    f"Failed to download {cfg.product_name} for year {year}: {e}"
                )
                continue

        # Skip aggregation if aggregated file already exists
        if cfg.agg == "fwi_agg":
            if agg_file.exists():
                logger.info(f"Aggregated file already exists, skipping: {agg_file}")
            else:
                print(f"Aggregating {cfg.product_name} for year {year}")
                try:
                    fwi_agg(
                        raw_file,
                        year,
                        cfg.temp_offset,
                        res=0.25,
                        store=True,
                    )
                    logger.info(f"Successfully aggregated year {year}")
                except Exception as e:
                    logger.error(f"Failed to aggregate year {year}: {e}")


if __name__ == "__main__":
    cds_wrapper()
