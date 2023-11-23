import pandas as pd
import numpy as np
import netCDF4 as nc
import os
import geopandas
import plotly.express as px

# select which year you want to observe
year = 2016

# feature we want to visualilze
feature = 'Rainf_f'

nc4file = 'NCALDAS_NOAH0125_D.A20160902.002.nc4'
path = "./unsorted_data/NCALDAS_data/" + str(year)

current_directory = os.getcwd()


ds = nc.Dataset(os.path.join(path,nc4file), 'r')
lat, lon = ds.variables['lat'][:], ds.variables['lon'][:]
    
xx, yy = np.meshgrid(lon, lat)

xx = np.ravel(xx)
yy = np.ravel(yy)

print(xx.shape)

df = pd.DataFrame()

df[feature] = np.ravel(ds.variables[feature][:])
df['lon'] = xx
df['lat'] = yy

ds.close()

df = df.dropna()

print(df.head())
print(df[feature].min())
print(df[feature].max())


print(df.columns)

fig = px.scatter_mapbox(df, lat=df['lat'], lon=df['lon'], color=df[feature],
                           color_continuous_scale="Bluered",
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5, 
                          )

fig.update_layout(
    font=dict(
        family="Arial Black",
        size=18,  # Set the font size here
    )
)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()