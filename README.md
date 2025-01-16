Overview
--------

rslearn_projects contains the training datasets, model weights, and corresponding code
for machine learning applications built on top of
[rslearn](https://github.com/allenai/rslearn/) at Ai2.

- **Model weights and Code**: Licensed under [Apache License 2.0](LICENSE).
- **Annotations**: Licensed under [CC-BY 4.0](DATA_LICENSE).


Setup
-----

Install rslearn:

    git clone https://github.com/allenai/rslearn.git
    cd rslearn
    pip install .[extra]

Install requirements:

    cd ..
    git clone https://github.com/allenai/rslearn_projects.git
    cd rslearn_projects
    pip install -r requirements.txt

rslearn_projects includes tooling that expects model checkpoints and auxiliary files to
be stored in an `RSLP_PREFIX` directory. Create a file `.env` to set the `RSLP_PREFIX`
environment variable:

    mkdir project_data
    echo "RSLP_PREFIX=project_data/" > .env


Applications
------------

- [Sentinel-2 Vessel Detection](docs/sentinel2_vessels.md)
- [Sentinel-2 Vessel Attribute Prediction](docs/sentinel2_vessel_attribute.md)
- [Landsat Vessel Detection](docs/landsat_vessels.md)
- [Satlas: Solar Farm Segmentation](docs/satlas_solar_farm.md)
- [Satlas: Wind Turbine Detection](docs/satlas_wind_turbine.md)
- [Satlas: Marine Infrastructure Detection](docs/satlas_marine_infra.md)
- [Forest Loss Driver Classification](docs/forest_loss_driver.md)
- [Maldives Ecosystem Mapping](docs/maldives_ecosystem_mapping.md)
