"""Constants for forest loss driver classification."""

# The group used for all the rslearn windows.
GROUP = "default"

# File containing the "good" windows. These are the windows that have images for all of
# the layers available, and have a valid prediction for the driver category. Only these
# windows will be displayed in the web app.
WINDOWS_FNAME = "good_windows.json"

# File containing the GeoJSON data constructed from the forest loss driver predictions.
GEOJSON_FNAME = "forest_loss_events.geojson"

# Where to store tiles. These need to be in a publicly accessible bucket.
DEFAULT_TILE_PATH = "gs://ai2-rslearn-projects-data/forest_loss_driver/tiles/"

# A special file indicating that this dataset is ready to serve from the web app.
READY_FOR_SERVING_FNAME = "ready_for_serving"
