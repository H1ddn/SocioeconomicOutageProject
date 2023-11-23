import json
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

column_to_plot = 'diff'

gdf_geojson = gpd.read_file('geojson-counties-fips.json')

print(gdf_geojson)

gdf_geojson = gdf_geojson.rename(columns={'id': 'FIPS'})

df = pd.read_csv('sorted_data/iso_2016_from_2015.csv', converters={'FIPS' : str})

df['diff'] = df['thresh'] - df['iso']

merged_gdf = gdf_geojson.merge(df, on='FIPS', how='left')

min_value = merged_gdf[column_to_plot].min()
max_value = merged_gdf[column_to_plot].max()

grouped = merged_gdf.groupby('DATE')

def update(frame):
    plt.clf()  # Clear the current plot
    data = grouped.get_group(frame)
    # Plot your data for the given time step, e.g., using geopandas' plot method.
    data.plot(column=column_to_plot, cmap='coolwarm', vmin=min_value, vmax=max_value, legend=True, ax=plt.gca())
    plt.title(f'Date: {frame}')

# Create the animation
animation = FuncAnimation(plt.gcf(), update, frames=grouped.groups.keys(), repeat=False)

animation.save('choropleth_timelapses/diff_iso_2016.mp4', writer='ffmpeg', fps=10)  # Adjust 'fps' as needed