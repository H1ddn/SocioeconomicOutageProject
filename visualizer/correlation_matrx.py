import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # for data visualiztions

df = pd.read_csv('sorted_data/2016/data_2016.csv', converters={'FIPS' : str})

feature_and_target_names = ['outage_max', 'outage_avg', 'outage_duration', 
                            'outage_max_3d', 'outage_avg_3d', 'outage_duration_3d',
                            'Pov_pct','Med_inc', 'Unemp_rate', 'Soc_vuln', 'Comm_res',
                            'Hurricane_score', 'River_flood_score', 'Coast_flood_score', 'Tair_f',  
                            'Qair_f', 'Psurf_f', 'Wind_f', 'Rainf_f', 'FloodedFrac', 'Qs', 'Qsb','Streamflow', 
                            'Wind_f_3d','Rainf_f_3d','Qs_3d','Qsb_3d',
                            'Res_customers']

X = df[feature_and_target_names]

fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches

# obtain a correlation heatmap 
sns.heatmap(X.corr(), annot=False, linewidths=0.05, ax=ax, fmt='.2f')
plt.show()