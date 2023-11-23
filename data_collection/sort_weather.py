import pandas as pd
import glob
import os
from datetime import datetime

year = '2015'

# Get all the csv files in directory
csv_files = glob.glob("unsorted_data/NCALDAS_bycounty/" + year + "/*.csv")

# Create a list of dataframes
dfs = []

# Iterate over the csv files
for csv_file in csv_files:

    date = os.path.basename(csv_file)[0:8]

    date = datetime.strptime(date, "%Y%m%d")

    # Read the csv file into a dataframe
    df = pd.read_csv(csv_file, converters={'properties.GEO_ID' : str})

    df.rename(columns={'properties.GEO_ID': 'FIPS'}, inplace=True)

    df['DATE'] = date

    # Append the dataframe to the list of dataframes
    dfs.append(df)

# Concatenate the dataframes into one dataframe
df = pd.concat(dfs)

df = df[df.Tair_f_tot != 0]

df.drop(columns=['Tair_f_tot', 'Qair_f_tot', 'Psurf_f_tot', 'Wind_f_tot', 'Rainf_f_tot', 'FloodedFrac_tot', 'Qs_tot', 'Qsb_tot', 'Streamflow_tot'], inplace=True)

df.to_csv('sorted_data/' + year +'/NCALDAS_daily_' + year +'.csv', index = False)
