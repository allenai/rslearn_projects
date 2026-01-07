### Explore the available data sources for landslides.
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

from utils import parse_name_as_date


data_root = '/weka/dfive-default/piperw/data/landslide/'


# Kaggle Global Landslide Catalog (GLC)
# Source: https://www.kaggle.com/datasets/nafayunnoor/global-landslide-catalog-glc-dataset/data
# Download: curl -L -o global-landslide-catalog-glc-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nafayunnoor/global-landslide-catalog-glc-dataset
# Data: landslide datapoints each with event date, lat/lon coordinate, trigger; 2-3K events 2015-2017, the rest are older
glc_csv_path = os.path.join(data_root, 'glc/Global_Landslide_Catalog_Export_20250201.csv')
df = pd.read_csv(
    glc_csv_path,
    encoding='utf-8',
    lineterminator='\n',  # Handle ^M line endings
)

print('\n=== SAMPLE EVENTS - raw ===')
print(df.head(10))

df.columns = df.columns.str.replace('^M', '', regex=False)
df.columns = df.columns.str.strip()

df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

string_columns = ['event_id', 'event_time', 'location_description', 'location_accuracy']
for col in string_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('^M', '', regex=False)
        df[col] = df[col].str.strip()

columns_of_interest = [
    'event_id', 
    'event_date', 
    'event_time', 
    'location_description', 
    'location_accuracy', 
    'latitude', 
    'longitude'
]

filtered_df = df[columns_of_interest].copy()
print('\n=== SAMPLE EVENTS - filtered ===')
print(filtered_df.head(10))

df_with_year = df.copy()
df_with_year['year'] = df_with_year['event_date'].dt.year
df_with_year = df_with_year.dropna(subset=['year'])

events_per_year = df_with_year['year'].value_counts().sort_index()
print('\n=== EVENTS PER YEAR ===')
print(events_per_year)


# Nepal Landslide Fatality Database (Petley et al.)
# Source: https://pubs.geoscienceworld.org/gsl/books/edited-volume/1519/chapter-abstract/107200818/On-the-impact-of-urban-landslides?redirectedFrom=fulltext
# Data: landslides from 1978-2006, too old to get overlapping S1/S2 data.


# High Mountain Asia Multi-Temporal Landslide Inventories (NSIDC)
# Source: https://nsidc.org/data/hma_mtli/versions/1?utm_source=chatgpt.com#anchor-data-access-tools
# Data: polygons of landslide footprints from 2009-12-01 to 2018-12-31 but no specific landslide event dates
shp_path = os.path.join(data_root, 'nsidc/HMA_MTLI_1-20260106_221858/HMA_MTLI_v01_landslide_inventory/HMA_MTLI_v01_Footprint.shp')
gdf = gpd.read_file(shp_path)
print(gdf.crs)
print(gdf.columns)
print(gdf.head())


# Regional inventories via ICIMOD
# Source: https://rds.icimod.org/Home/DataDetail?metadataId=31016&utm_source=chatgpt.com
# Data: polygons of landslide footprints in Nepal from 2001->2016, ~1k-2500 events in 2015-2016
shp_path = os.path.join(data_root, 'icimod/data/14dist_ls.shp')
gdf = gpd.read_file(shp_path)
print(gdf.crs)
print(gdf.columns)
print(gdf.head())

gdf['centroid'] = gdf.geometry.centroid
gdf['latitude'] = gdf['centroid'].y
gdf['longitude'] = gdf['centroid'].x
gdf['date'] = gdf['Name'].apply(parse_name_as_date)

filtered_gdf = gdf[['date', 'latitude', 'longitude']].copy()
print('\n=== SAMPLE EVENTS - filtered ===')
print(filtered_gdf.head())

filtered_gdf['date'] = pd.to_datetime(filtered_gdf['date'], errors='coerce')
valid = filtered_gdf.dropna(subset=['date']).copy()
valid['year'] = valid['date'].dt.year
events_per_year = valid['year'].value_counts().sort_index()
print('\nNumber of landslide events per year:')
print(events_per_year)


# Far-Western Nepal Multi-Temporal Landslide Inventory
# Source: https://data.niaid.nih.gov/resources?id=zenodo_4290099&utm_source=chatgpt.com
# Data: center point and year of landslides from 1992-2018 with ~1-2k events 2015-2018
shp_path = os.path.join(
    data_root,
    'niaid',
    'LandslideInventory_FarWesternNepal',
    'LandslideInventory_FarWesternNepal_Points_Dated1992_2018.shp',
)
gdf = gpd.read_file(shp_path)

print(gdf.crs)
print(gdf.columns)
print(gdf.head())

gdf_wgs84 = gdf.to_crs(epsg=4326)
gdf_wgs84['latitude'] = gdf_wgs84.geometry.y
gdf_wgs84['longitude'] = gdf_wgs84.geometry.x
gdf_wgs84['year'] = gdf_wgs84['Year'].astype('Int64')

events = gdf_wgs84[['year', 'latitude', 'longitude']].copy()
print('\n=== Sample events (Far Western Nepal) ===')
print(events.head())

events_per_year = events['year'].value_counts().sort_index()
print('\nNumber of landslide events per year (Far Western Nepal dataset):')
print(events_per_year)


# Sen12Landslides
# Source: https://huggingface.co/datasets/paulhoehn/Sen12Landslides
# Data: pre_date & post_date of 75k landslide events with polygons from 2016-2021
shp_path = os.path.join(data_root, 'sen12landslides/inventories.shp')
gdf = gpd.read_file(shp_path)

print(gdf.crs)
print(gdf.columns)
print(gdf.head())

gdf["centroid"] = gdf.geometry.centroid
gdf["latitude"] = gdf["centroid"].y
gdf["longitude"] = gdf["centroid"].x
gdf["event_date"] = pd.to_datetime(gdf["event_date"], errors="coerce")

events = gdf[["id", "event_date", "latitude", "longitude", "location", "event_type", "geometry"]].copy()

print("\n=== SAMPLE EVENTS (Sen12Landslides inventories) ===")
print(events.head())

valid = events.dropna(subset=["event_date"]).copy()
valid["year"] = valid["event_date"].dt.year

events_per_year = valid["year"].value_counts().sort_index()
print("\nNumber of landslide polygons per year:")
print(events_per_year)