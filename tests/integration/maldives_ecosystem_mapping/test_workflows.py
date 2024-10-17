import os

from rslp.maldives_ecosystem_mapping.data_pipeline import (
    DataPipelineConfig,
    data_pipeline,
)


def test_workflows(tmp_path):
    dp_config = DataPipelineConfig(
        ds_root=str(tmp_path),
        src_dir="tests/integration/maldives_ecosystem_mapping/data",
        islands_fname="tests/integration/maldives_ecosystem_mapping/data/islands.json",
    )
    data_pipeline(dp_config)
    assert os.path.exists(
        os.path.join(
            tmp_path,
            "windows/crops/fake_2024-01-01-00-00_0_0/layers/maxar/R_G_B/geotiff.tif",
        )
    )
