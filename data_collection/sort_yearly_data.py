import pandas as pd
import numpy as np

year = 2015

df = pd.read_excel('unsorted_data/energy_data/2016cityandcountyenergyprofiles.xlsb', engine='pyxlsb')

df = df.iloc[4:, :]
df = df.reset_index(drop=True)

# ['FIPS', 'Res_elec', 'Res_customers', 'Cml_elec', 'Cml_customers']

sorteddf = pd.DataFrame()

sorteddf['FIPS'] = df.iloc[:, 3]
sorteddf['FIPS'] = sorteddf['FIPS'].astype(str)
sorteddf['FIPS'] = sorteddf['FIPS'].str.zfill(5)
sorteddf['Res_elec'] = df.iloc[:, 14]
sorteddf['Res_customers'] = df.iloc[:, 18]
sorteddf['Cml_elec'] = df.iloc[:, 25]
sorteddf['Cml_customers'] = df.iloc[:, 31]

df = pd.read_csv('unsorted_data/NRI_data/NRI_Table_Counties.csv')

df['STCOFIPS'] = df['STCOFIPS'].astype(str)
df['STCOFIPS'] = df['STCOFIPS'].str.zfill(5)

# ['geoloc', 'COUNTY', 'STATE', 'Soc_vuln', 'Comm_res', 'Tornado_score', 'Hurricane_score', 'River_flood_score', 'Coast_flood_score']

sorteddf['geoloc'] = 'nan'
sorteddf['COUNTY'] = 'nan'
sorteddf['STATE'] = 'nan'
sorteddf['Soc_vuln'] = 0.00
sorteddf['Comm_res'] = 0.00
sorteddf['Tornado_score'] = 0.00
sorteddf['Hurricane_score'] = 0.00
sorteddf['River_flood_score'] = 0.00
sorteddf['Coast_flood_score'] = 0.00

for ind in df.index:
    if(pd.isnull(df.at[ind, 'COUNTYTYPE'])):
        county = df.loc[ind, 'COUNTY']
        state = df.loc[ind, 'STATE']
    else:
        county = df.loc[ind, 'COUNTY'] + ' ' + df.loc[ind, 'COUNTYTYPE']
        state = df.loc[ind, 'STATE']

    sortindx = sorteddf.loc[sorteddf['FIPS'] == df.loc[ind, 'STCOFIPS']].index 
    sorteddf.loc[sortindx, 'geoloc'] = '.' + county.lower() + ', ' + state.lower()
    sorteddf.loc[sortindx, 'COUNTY'] = county
    sorteddf.loc[sortindx, 'STATE'] = state
    sorteddf.loc[sortindx, 'Soc_vuln'] = df.loc[ind, 'SOVI_SCORE']
    sorteddf.loc[sortindx, 'Comm_res'] = df.loc[ind, 'RESL_SCORE']
    sorteddf.loc[sortindx, 'Tornado_score'] = df.loc[ind, 'TRND_RISKS']
    sorteddf.loc[sortindx, 'Hurricane_score'] = df.loc[ind, 'HRCN_RISKS']
    sorteddf.loc[sortindx, 'River_flood_score'] = df.loc[ind, 'RFLD_RISKS']
    sorteddf.loc[sortindx, 'Coast_flood_score'] = df.loc[ind, 'CFLD_RISKS']    

sorteddf = sorteddf.drop(sorteddf.loc[sorteddf['geoloc'] == 'nan'].index)

df = pd.read_excel('unsorted_data\census_data\est' + str(year - 2000) + 'all.xls')

df = df.iloc[3:, :]
df = df.reset_index(drop=True)
df.iloc[:, 0] = df.iloc[:, 0].astype(str)
df.iloc[:, 1] = df.iloc[:, 1].astype(str)
df['FIPS'] = df.iloc[:, 0] + df.iloc[:, 1]

# ['Pov_pct', 'Med_inc']
sorteddf['Pov_pct'] = 0.00
sorteddf['Med_inc'] = 0.00

pd.options.mode.chained_assignment = None

for ind in df.index:
    sorteddf['Pov_pct'].loc[sorteddf['FIPS'] == df.loc[ind, 'FIPS']] = df.iloc[ind, 4]
    sorteddf['Med_inc'].loc[sorteddf['FIPS'] == df.loc[ind, 'FIPS']] = df.iloc[ind, 22]

df = pd.read_excel("unsorted_data\census_data\co-est2019-annres.xlsx")

df = df.iloc[3:, :]
df = df.reset_index(drop=True)

sorteddf['Pop_est'] = 0

for ind in df.index:
    sorteddf['Pop_est'].loc[sorteddf['geoloc'] == df.iloc[ind, 0].lower()] = df.iloc[ind, year - 2007]

df = pd.read_excel('unsorted_data/unemp_rate_data/Unemployment.xlsx')

df = df.iloc[4:, :]
df = df.reset_index(drop=True)
df.iloc[:, 0] = df.iloc[:, 0].astype(str)

sorteddf['Unemp_rate'] = 0.0

for ind in df.index:
    sorteddf['Unemp_rate'].loc[sorteddf['FIPS'] == df.iloc[ind, 0]] = df.iloc[ind, (year - 1998)*4 + 1]

sorteddf = sorteddf.drop('geoloc', axis=1)
sorteddf.to_csv('sorted_data/' + str(year) + '/yearly_data_' + str(year) + '.csv', index=False)