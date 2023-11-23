import pandas as pd

year = '2015'

df = pd.read_csv('unsorted_data/outages_data/' + year + '/eaglei_outages_' + year + '.csv', converters={'fips_code': str})

countydf = pd.read_csv('sorted_data/energy_profile.csv', converters={'FIPS' : str})

df.sort_values(['fips_code','run_start_time'], inplace=True)
df.reset_index(drop=True, inplace=True)

df = df.rename(columns={'fips_code': 'FIPS'})

df['run_start_time']= pd.to_datetime(df['run_start_time'])

df['run_start_time'] = df['run_start_time'].dt.date

# get max value
maxdf = df.groupby(['FIPS','run_start_time'],as_index=False)['sum'].max().agg(list)
maxdf.rename( columns={'sum':'outage_max'}, inplace=True )
maxdf = pd.merge(maxdf, countydf[['FIPS','Res_customers']], on=['FIPS'], how='left')
maxdf['outage_max'] = maxdf['outage_max'] / maxdf['Res_customers']
maxdf.drop('Res_customers', axis=1, inplace=True)

# get average value
avgdf = df.groupby(['FIPS','run_start_time'],as_index=False)['sum'].sum().agg(list)
avgdf['sum'] = avgdf['sum'] / 96 # there are 96 15-minute sections in a day
avgdf.rename( columns={'sum':'outage_avg'}, inplace=True )
avgdf = pd.merge(avgdf, countydf[['FIPS','Res_customers']], on=['FIPS'], how='left')
avgdf['outage_avg'] = avgdf['outage_avg'] / avgdf['Res_customers']
avgdf.drop('Res_customers', axis=1, inplace=True)

df = df.loc[df['sum'] >= df['sum'].mean()]

countdf = df.groupby(['FIPS', 'run_start_time'],as_index=False).size().agg(list)

maxdf = pd.merge(maxdf, avgdf, on=['FIPS','run_start_time'], how='left')
df = pd.merge(maxdf, countdf, on=['FIPS','run_start_time'], how='left')

df['size'].fillna(0, inplace=True)

df['size'] = df['size'] / 100

df = df.rename(columns={'run_start_time': 'DATE', 'size' : 'outage_duration'})

df.to_csv('sorted_data/' + year + '/outages_' + year + '.csv', index = False)