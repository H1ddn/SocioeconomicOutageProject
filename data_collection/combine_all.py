import pandas as pd

year = '2015'

countydf = pd.read_csv('sorted_data/' + year + '/yearly_data_' + year + '.csv', converters={'FIPS' : str})

outagedf = pd.read_csv('sorted_data/' + year + '/outages_' + year + '.csv', converters={'FIPS' : str})

weatherdf = pd.read_csv('sorted_data/' + year + '/NCALDAS_daily_' + year + '.csv', converters={'FIPS' : str})

df = pd.merge(weatherdf, outagedf, on=['FIPS','DATE'], how='left')

df['outage_avg'].fillna(0, inplace=True)
df['outage_max'].fillna(0, inplace=True)
df['outage_duration'].fillna(0, inplace=True)

df = pd.merge(df, countydf, on=['FIPS'], how='left')

df = df[df['Res_elec'].notna()]

print(df.isna().sum())

df.fillna(-1, inplace=True)

print(df.shape)
print(df.head())

# sort by fips, date
df.sort_values(by=['FIPS', 'DATE'], inplace=True)

df = df.reindex(columns=[
                    'FIPS', 'COUNTY', 'STATE', 'DATE', # metadata
                    'outage_avg', 'outage_max', 'outage_duration', # potential target values
                    'Pov_pct', 'Med_inc', 'Pop_est', 'Unemp_rate', # socioeconomic data
                    'Soc_vuln', 'Comm_res', 'Tornado_score', 'Hurricane_score', 'River_flood_score', 'Coast_flood_score', # FEMA scores
                    'Tair_f', 'Qair_f', 'Psurf_f', 'Wind_f', 'Rainf_f', 'FloodedFrac', 'Qs', 'Qsb', 'Streamflow', # weather data
                    'Res_elec', 'Res_customers', 'Cml_elec', 'Cml_customers', # electric profiles
                    ])

df.to_csv('sorted_data/' + year + '/raw_' + year + '.csv', index=False)