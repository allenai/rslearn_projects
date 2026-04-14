"""Export the 2024 MapBiomas Brazil land-cover layer from Earth Engine."""

import os
import time

import ee

ee.Authenticate()
ee.Initialize(project=os.environ["GCP_PROJECT"])

mapbiomas = ee.Image(
    "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_coverage_v2"
)

image_2024 = mapbiomas.select("classification_2024").toInt16()

brazil = (
    ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
    .filter(ee.Filter.eq("ADM0_NAME", "Brazil"))
    .geometry()
)

task = ee.batch.Export.image.toDrive(
    image=image_2024,
    description="mapbiomas_brazil_collection10_2024_30m",
    folder="earthengine",
    fileNamePrefix="mapbiomas_brazil_collection10_2024_30m",
    region=brazil,
    scale=30,
    maxPixels=3e10,
    fileFormat="GeoTIFF",
)

task.start()

while task.active():
    print("Task running...", task.status().get("state"))
    time.sleep(30)

print("Final status:", task.status())
