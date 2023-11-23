import pandas as pd
import json
import plotly.express as px
import numpy as np

feature='Tornado_score'

with open('geojson-counties-fips.json') as response:
    counties = json.load(response)

df = pd.read_csv('./sorted_data/2016/raw_2016.csv', dtype={'FIPS': str})

df = df.loc[df['DATE'] == '2016-09-02']

print(df[feature].max())

fig = px.choropleth_mapbox(df, geojson=counties, locations='FIPS', color=feature,
                           color_continuous_scale="Bluered",
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5
                          )

fig.update_layout(
    font=dict(
        family="Arial Black",
        size=18,  # Set the font size here
    )
)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
