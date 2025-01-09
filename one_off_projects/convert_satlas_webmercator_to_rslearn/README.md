About
-----

This project is for converting the various Satlas application training datasets (wind
turbines, solar farms, marine infrastructure) to rslearn format.

The conversion has completed so it is now in one_off_projects and the code does not
need to be maintained anymore.


Wind Turbines
-------------

The wind turbine training data consists of two components:

(1) Labels in siv.
(2) Additional points derived from NAIP wind turbine detection model applied on many
    NAIP images.

The second should be run after the first to only add the remaining points in the
dataset that aren't from labels in siv.

    python wind_turbine/convert_siv_labels.py
    python
