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

        logger.info(f"Downloading CDS collection {cfg.product_name} for year {year}")

        try:
            output_file = str(output_dir / f"{cfg.product_name}-{year}.nc")

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
                    "month": [
                        "01",
                        "02",
                        "03",
                        "04",
                        "05",
                        "06",
                        "07",
                        "08",
                        "09",
                        "10",
                        "11",
                        "12",
                    ],
                    "day": [
                        "01",
                        "02",
                        "03",
                        "04",
                        "05",
                        "06",
                        "07",
                        "08",
                        "09",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                        "21",
                        "22",
                        "23",
                        "24",
                        "25",
                        "26",
                        "27",
                        "28",
                        "29",
                        "30",
                        "31",
                    ],
                },
                output_file,
            )

            logger.info(f"Successfully downloaded {output_file}")
            if cfg.agg == "fwi_agg":
                print(f"Aggregating {cfg.product_name} for year {year}")
                fwi_agg(
                    output_dir / f"{cfg.product_name}-{year}.nc",
                    year,
                    cfg.temp_offset,
                    res=0.25,
                    store=True,
                )

        except Exception as e:
            logger.error(
                f"Failed to download CDS collection {cfg.product_name} for year {year}"
            )
            logger.error(e)


if __name__ == "__main__":
    cds_wrapper()
