from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
import pandas as pd
import numpy as np
import netCDF4 as nc
import os
import sys
from rtree import index

# select which year you want to observe
year = 2015

# features we want to collect
features = ['Tair_f', 'Qair_f', 'Psurf_f', 'Wind_f', 'Rainf_f', 'FloodedFrac', 'Qs', 'Qsb', 'Streamflow']


path = "./unsorted_data/NCALDAS_data/" + str(year)

# gather important coordinates
df = pd.read_json("gz_2010_us_050_00_5m.json", encoding='latin-1')
df = pd.json_normalize(df['features'])

'''
Southeast_states = ['Louisiana','Mississippi','Alabama',
                  'Florida','Georgia','South Carolina',
                  'North Carolina','Virginia']
'''

Southeast_states = ['22','28','01',
                  '12','13','45',
                  '37','51']

df = df.loc[df['properties.STATE'].isin(Southeast_states)]

df["properties.GEO_ID"] = df["properties.GEO_ID"].str[-5:]
df = df[['properties.GEO_ID','geometry.type','geometry.coordinates',]]


# turn coords into shapely polygons
polys = []
i = 0
for ind in df.index:
    if(df['geometry.type'][ind] == 'Polygon'):
        polys.append(Polygon(df['geometry.coordinates'][ind][0]))
        df['geometry.coordinates'][ind] = [i]
        i += 1
    elif(df['geometry.type'][ind] == 'MultiPolygon'):
        indexes = []
        for coord in df['geometry.coordinates'][ind][0]:
            polys.append(Polygon(coord))
            indexes.append(i)
            i += 1
        df['geometry.coordinates'][ind] = indexes

# create RTree from polygons
idx = index.Index()
for pos, poly in enumerate(polys):
       idx.insert(pos, shape(poly).bounds)

print('hi')

current_directory = os.getcwd()

for file in os.listdir(path):

    # add/reset features
    for feature in features:
        df[feature] = 0.0000
        df[feature + '_tot'] = 0


    ds = nc.Dataset(os.path.join(path,file), 'r')
    lat, lon = ds.variables['lat'][:], ds.variables['lon'][:]
    
    feats = dict()
    for feature in features:
        feats.update({feature : ds.variables[feature][:]})

    ds.close()

    xx, yy = np.meshgrid(lon, lat)
    points = np.column_stack((xx.ravel(), yy.ravel()))

    for x in range(224):
        for y in range(464):
            point = shape(Point(xx[x][y],yy[x][y]))
            # iterate through spatial index
            for j in idx.intersection(point.coords[0]):
                if point.within(shape(polys[j])):
                        for ind in df.index:
                            if j in df['geometry.coordinates'][ind]:

                                for feature in features:
                                    featot = feature + '_tot'

                                    if(feats[feature][0][x][y] > -999):
                                        df.loc[ind, feature] += feats[feature][0][x][y]
                                        df.loc[ind, featot] += 1 

    for ind in df.index:

        for feature in features:
            featot = feature + '_tot'

            if(df.loc[ind, featot] > 0):
                df.loc[ind, feature] = df.loc[ind, feature] / df.loc[ind, featot]


    csvdf = df.drop('geometry.coordinates',axis=1)

    print(csvdf.sort_values(by='Tair_f', ascending=False))

    csvdf.to_csv(os.path.join(current_directory,"unsorted_data", "NCALDAS_bycounty",str(year),(file[-16:-8] + '.csv')), index=False)

    print(file[-16:-8])