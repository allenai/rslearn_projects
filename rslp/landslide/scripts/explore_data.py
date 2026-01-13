"""
Explore the available data sources for landslides.
"""
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image

from utils import parse_name_as_date


data_root = '/weka/dfive-default/piperw/data/landslide/'


# # Kaggle Global Landslide Catalog (GLC)
# # Source: https://www.kaggle.com/datasets/nafayunnoor/global-landslide-catalog-glc-dataset/data
# # Download: curl -L -o global-landslide-catalog-glc-dataset.zip https://www.kaggle.com/api/v1/datasets/download/nafayunnoor/global-landslide-catalog-glc-dataset
# # Data: landslide datapoints each with event date, lat/lon coordinate, trigger; 2-3K events 2015-2017, the rest are older
# glc_csv_path = os.path.join(data_root, 'glc/Global_Landslide_Catalog_Export_20250201.csv')
# df = pd.read_csv(
#     glc_csv_path,
#     encoding='utf-8',
#     lineterminator='\n',  # Handle ^M line endings
# )

# print('\n=== SAMPLE EVENTS - raw ===')
# print(df.head(10))

# df.columns = df.columns.str.replace('^M', '', regex=False)
# df.columns = df.columns.str.strip()

# df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

# df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
# df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# string_columns = ['event_id', 'event_time', 'location_description', 'location_accuracy']
# for col in string_columns:
#     if col in df.columns:
#         df[col] = df[col].astype(str).str.replace('^M', '', regex=False)
#         df[col] = df[col].str.strip()

# columns_of_interest = [
#     'event_id', 
#     'event_date', 
#     'event_time', 
#     'location_description', 
#     'location_accuracy', 
#     'latitude', 
#     'longitude'
# ]

# filtered_df = df[columns_of_interest].copy()
# print('\n=== SAMPLE EVENTS - filtered ===')
# print(filtered_df.head(10))

# df_with_year = df.copy()
# df_with_year['year'] = df_with_year['event_date'].dt.year
# df_with_year = df_with_year.dropna(subset=['year'])

# events_per_year = df_with_year['year'].value_counts().sort_index()
# print('\n=== EVENTS PER YEAR ===')
# print(events_per_year)


# # Nepal Landslide Fatality Database (Petley et al.)
# # Source: https://pubs.geoscienceworld.org/gsl/books/edited-volume/1519/chapter-abstract/107200818/On-the-impact-of-urban-landslides?redirectedFrom=fulltext
# # Data: landslides from 1978-2006, too old to get overlapping S1/S2 data.


# # High Mountain Asia Multi-Temporal Landslide Inventories (NSIDC)
# # Source: https://nsidc.org/data/hma_mtli/versions/1?utm_source=chatgpt.com#anchor-data-access-tools
# # Data: polygons of landslide footprints from 2009-12-01 to 2018-12-31 but no specific landslide event dates
# shp_path = os.path.join(data_root, 'nsidc/HMA_MTLI_1-20260106_221858/HMA_MTLI_v01_landslide_inventory/HMA_MTLI_v01_Footprint.shp')
# gdf = gpd.read_file(shp_path)
# print(gdf.crs)
# print(gdf.columns)
# print(gdf.head())


# # Regional inventories via ICIMOD
# # Source: https://rds.icimod.org/Home/DataDetail?metadataId=31016&utm_source=chatgpt.com
# # Data: polygons of landslide footprints in Nepal from 2001->2016, ~1k-2500 events in 2015-2016
# shp_path = os.path.join(data_root, 'icimod/data/14dist_ls.shp')
# gdf = gpd.read_file(shp_path)
# print(gdf.crs)
# print(gdf.columns)
# print(gdf.head())

# gdf['centroid'] = gdf.geometry.centroid
# gdf['latitude'] = gdf['centroid'].y
# gdf['longitude'] = gdf['centroid'].x
# gdf['date'] = gdf['Name'].apply(parse_name_as_date)

# filtered_gdf = gdf[['date', 'latitude', 'longitude']].copy()
# print('\n=== SAMPLE EVENTS - filtered ===')
# print(filtered_gdf.head())

# filtered_gdf['date'] = pd.to_datetime(filtered_gdf['date'], errors='coerce')
# valid = filtered_gdf.dropna(subset=['date']).copy()
# valid['year'] = valid['date'].dt.year
# events_per_year = valid['year'].value_counts().sort_index()
# print('\nNumber of landslide events per year:')
# print(events_per_year)


# # Far-Western Nepal Multi-Temporal Landslide Inventory
# # Source: https://data.niaid.nih.gov/resources?id=zenodo_4290099&utm_source=chatgpt.com
# # Data: center point and year of landslides from 1992-2018 with ~1-2k events 2015-2018
# shp_path = os.path.join(
#     data_root,
#     'niaid',
#     'LandslideInventory_FarWesternNepal',
#     'LandslideInventory_FarWesternNepal_Points_Dated1992_2018.shp',
# )
# gdf = gpd.read_file(shp_path)

# print(gdf.crs)
# print(gdf.columns)
# print(gdf.head())

# gdf_wgs84 = gdf.to_crs(epsg=4326)
# gdf_wgs84['latitude'] = gdf_wgs84.geometry.y
# gdf_wgs84['longitude'] = gdf_wgs84.geometry.x
# gdf_wgs84['year'] = gdf_wgs84['Year'].astype('Int64')

# events = gdf_wgs84[['year', 'latitude', 'longitude']].copy()
# print('\n=== Sample events (Far Western Nepal) ===')
# print(events.head())

# events_per_year = events['year'].value_counts().sort_index()
# print('\nNumber of landslide events per year (Far Western Nepal dataset):')
# print(events_per_year)


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


# Plot landslide locations per year and create GIF animation

# Create output directory if it doesn't exist
os.makedirs('landslide_plots_by_year', exist_ok=True)

# Get the valid events with dates
valid_events = events.dropna(subset=["event_date"]).copy()
valid_events["year"] = valid_events["event_date"].dt.year

# Get unique years and sort them
years = sorted(valid_events['year'].unique())

# Calculate global min/max for consistent axes across all plots
lon_min = valid_events['longitude'].min()
lon_max = valid_events['longitude'].max()
lat_min = valid_events['latitude'].min()
lat_max = valid_events['latitude'].max()

# Add a small padding (5% of range)
lon_padding = (lon_max - lon_min) * 0.05
lat_padding = (lat_max - lat_min) * 0.05

print(f"Creating plots for {len(years)} years...")
print(f"Longitude range: [{lon_min:.2f}, {lon_max:.2f}]")
print(f"Latitude range: [{lat_min:.2f}, {lat_max:.2f}]")

# Store filenames for GIF creation
image_files = []

# Create a plot for each year
for year in years:
    # Filter data for this year
    year_data = valid_events[valid_events['year'] == year]
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot landslides for this year
    scatter = ax.scatter(
        year_data['longitude'], 
        year_data['latitude'],
        c='red',
        alpha=0.5,
        s=20,
        edgecolors='darkred',
        linewidths=0.3,
        label='Landslides'
    )
    
    # Set consistent axis limits
    ax.set_xlim(lon_min - lon_padding, lon_max + lon_padding)
    ax.set_ylim(lat_min - lat_padding, lat_max + lat_padding)
    
    # Add labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Global Landslide Events in {year}\nSen12Landslides Dataset', 
                 fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend with count
    ax.legend([f'Landslides (n={len(year_data):,})'], loc='upper right')
    
    # Equal aspect ratio for proper geographic display
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save the figure
    filename = f'landslide_plots_by_year/landslides_{year}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved {filename} ({len(year_data):,} events)")
    
    image_files.append(filename)
    
    plt.close()

print("\nAll plots saved to 'landslide_plots_by_year/' directory")

# Create GIF from the images
print("\nCreating GIF animation...")
images = [Image.open(img) for img in image_files]

# Save as GIF
gif_filename = 'landslides_animation.gif'
images[0].save(
    gif_filename,
    save_all=True,
    append_images=images[1:],
    duration=1000,  # milliseconds per frame (1000ms = 1 second)
    loop=0  # 0 means loop forever
)

print(f"GIF saved as '{gif_filename}'")
print(f"  - {len(images)} frames")

# Get unique coordinates
coords = valid_events[['latitude', 'longitude']].values

print(f"Total landslide events: {len(coords):,}")

# Method 1: Exact duplicates (same lat/long to full precision)
exact_duplicates = valid_events.duplicated(subset=['latitude', 'longitude'], keep=False)
n_exact_duplicates = exact_duplicates.sum()
print(f"\nExact duplicate coordinates: {n_exact_duplicates:,}")

unique_coords_exact = valid_events.drop_duplicates(subset=['latitude', 'longitude'])
print(f"Unique coordinates (exact): {len(unique_coords_exact):,}")

# Method 2: Nearby duplicates within a distance threshold
# Distance threshold in degrees (approximate)
# 0.001 degrees ≈ 111 meters at the equator
# 0.0001 degrees ≈ 11 meters
# 0.00001 degrees ≈ 1.1 meters

threshold_deg = 0.001  # ~111 meters

print(f"\nChecking for nearby duplicates within {threshold_deg} degrees (~{threshold_deg*111:.1f} km)...")

# This can be slow for large datasets, so let's sample if needed
if len(coords) > 10000:
    print(f"  (Sampling 10,000 points for performance...)")
    sample_indices = np.random.choice(len(coords), 10000, replace=False)
    coords_sample = coords[sample_indices]
else:
    coords_sample = coords

# Compute pairwise distances
distances = cdist(coords_sample, coords_sample, metric='euclidean')

# Count pairs within threshold (excluding diagonal)
np.fill_diagonal(distances, np.inf)
nearby_pairs = np.sum(distances < threshold_deg, axis=1)

n_with_nearby = np.sum(nearby_pairs > 0)
max_nearby = np.max(nearby_pairs)

print(f"  Points with nearby neighbors: {n_with_nearby:,} / {len(coords_sample):,}")
print(f"  Maximum nearby neighbors for a single point: {max_nearby}")

# Find the point with most neighbors
if max_nearby > 0:
    max_idx = np.argmax(nearby_pairs)
    max_coord = coords_sample[max_idx]
    print(f"  Location with most neighbors: lat={max_coord[0]:.4f}, lon={max_coord[1]:.4f}")
    
# Method 3: Grid-based counting (faster for large datasets)
print(f"\nGrid-based analysis:")
# Round coordinates to grid cells
grid_resolution = 0.01  # degrees (~1.1 km)
valid_events['lat_grid'] = (valid_events['latitude'] / grid_resolution).round() * grid_resolution
valid_events['lon_grid'] = (valid_events['longitude'] / grid_resolution).round() * grid_resolution

# Count events per grid cell
grid_counts = valid_events.groupby(['lat_grid', 'lon_grid']).size().reset_index(name='count')
grid_counts_sorted = grid_counts.sort_values('count', ascending=False)

print(f"  Grid resolution: {grid_resolution}° (~{grid_resolution*111:.1f} km)")
print(f"  Total grid cells with events: {len(grid_counts):,}")
print(f"\nTop 10 grid cells with most events:")
print(grid_counts_sorted.head(10).to_string(index=False))

# Distribution of events per cell
print(f"\nDistribution of events per grid cell:")
print(f"  1 event: {(grid_counts['count'] == 1).sum():,} cells")
print(f"  2-5 events: {((grid_counts['count'] >= 2) & (grid_counts['count'] <= 5)).sum():,} cells")
print(f"  6-10 events: {((grid_counts['count'] >= 6) & (grid_counts['count'] <= 10)).sum():,} cells")
print(f"  11+ events: {(grid_counts['count'] > 10).sum():,} cells")
print(f"  Max events in one cell: {grid_counts['count'].max()}")