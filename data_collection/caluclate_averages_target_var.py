import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

year = '2015'

df = pd.read_csv('sorted_data/' + year + '/raw_' + year + '.csv', converters={'FIPS' : str})

# sort by fips, date
df.sort_values(by=['FIPS', 'DATE'], inplace=True)

# get averages by FIPS
df['outage_duration_3d'] = df.groupby('FIPS')['outage_duration'].rolling(window=7).mean().reset_index(level=0, drop=True)
df['outage_avg_3d'] = df.groupby('FIPS')['outage_avg'].rolling(window=7).mean().reset_index(level=0, drop=True)
df['outage_max_3d'] = df.groupby('FIPS')['outage_max'].rolling(window=7).mean().reset_index(level=0, drop=True)
df['Wind_f_3d'] = df.groupby('FIPS')['Wind_f'].rolling(window=7).mean().reset_index(level=0, drop=True)
df['Rainf_f_3d'] = df.groupby('FIPS')['Rainf_f'].rolling(window=7).mean().reset_index(level=0, drop=True)
df['FloodedFrac_3d'] = df.groupby('FIPS')['FloodedFrac'].rolling(window=7).mean().reset_index(level=0, drop=True)
df['Qs_3d'] = df.groupby('FIPS')['Qs'].rolling(window=7).mean().reset_index(level=0, drop=True)
df['Qsb_3d'] = df.groupby('FIPS')['Qsb'].rolling(window=7).mean().reset_index(level=0, drop=True)
df['Streamflow_3d'] = df.groupby('FIPS')['Streamflow'].rolling(window=7).mean().reset_index(level=0, drop=True)

# Shift the target variable back by one row
df['next_day_avg'] = df.groupby('FIPS')['outage_avg'].shift(-1).reset_index(level=0, drop=True)

# drop nan values (first 6 days of the year & last day of year)
df.dropna(subset=['Streamflow_3d'], inplace=True)
df.dropna(subset=['next_day_avg'], inplace=True)

# GET EAST COAST STATES ONLY (HURRICANES)
'''
Coastal_States = ['Texas','Louisiana','Mississippi','Alabama','Florida',
                  'Georgia','South Carolina','North Carolina','Virginia',
                  'District of Colombia','Maryland','Delaware','New Jersey'
                  'Pennsylvania','New York','Connecticut','Rhode Island',
                  'Vermont','New Hampshire','Maine']
'''              

Southeast_states = ['Louisiana','Mississippi','Alabama',
                  'Florida','Georgia','South Carolina',
                  'North Carolina','Virginia']

df = df.loc[df['STATE'].isin(Southeast_states)]

df = df.reindex(columns=[
                    'FIPS', 'COUNTY', 'STATE', 'DATE', # metadata
                    'next_day_avg', # target variable
                    'outage_avg', 'outage_max', 'outage_duration', # outage data values
                    'outage_avg_3d', 'outage_max_3d', 'outage_duration_3d', # outage data averages
                    'Pov_pct', 'Med_inc', 'Pop_est', 'Unemp_rate', # socioeconomic data
                    'Soc_vuln', 'Comm_res', 'Tornado_score', 'Hurricane_score', 'River_flood_score', 'Coast_flood_score', # FEMA scores
                    'Tair_f', 'Qair_f', 'Psurf_f', 'Wind_f', 'Rainf_f', 'FloodedFrac', 'Qs', 'Qsb', 'Streamflow', # weather data
                    'Wind_f_3d', 'Rainf_f_3d', 'FloodedFrac_3d', 'Qs_3d', 'Qsb_3d', 'Streamflow_3d', # weather data averages
                    'Res_elec', 'Res_customers', 'Cml_elec', 'Cml_customers', # electric profiles
                    ])

df.to_csv('sorted_data/' + year + '/data_' + year + '.csv', index=False)