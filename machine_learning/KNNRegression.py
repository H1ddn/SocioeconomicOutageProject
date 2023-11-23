import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from scipy.stats import zscore
from sklearn.model_selection import GridSearchCV
import statsmodels.stats.outliers_influence as infl
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

df = pd.read_csv('sorted_data/2016/data_2016.csv', converters={'FIPS' : str})

print(df['next_day_avg'].std())
print(df['next_day_avg'].mean())
print(df.shape)

feature_names = ['outage_max', 'outage_avg', 'outage_duration', 
                 'outage_max_3d', 'outage_avg_3d', 'outage_duration_3d',
                 'Pov_pct','Med_inc', 'Unemp_rate','Soc_vuln', 'Comm_res',
                 'Hurricane_score', 'River_flood_score', 'Coast_flood_score', 'Tair_f',  
                 'Qair_f', 'Psurf_f', 'Wind_f', 'Rainf_f', 'FloodedFrac', 'Qs', 'Qsb','Streamflow', 
                 'Res_customers',
                 'Wind_f_3d','Rainf_f_3d','Qs_3d','Qsb_3d']

X = df[feature_names]

y = df['next_day_avg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the hyperparameters and their possible values to search
param_grid = {
    'n_neighbors': [9, 11, 13, 15, 17],
    'weights': ['distance'],
    'p': [1],  # For Minkowski distance (1 for Manhattan, 2 for Euclidean)
    'algorithm': ['auto']
}

knn = KNeighborsRegressor(n_neighbors= 9, weights='distance', p=1, algorithm='auto')

# Fit the GridSearchCV to the training data
knn.fit(X_train_scaled, y_train)

# Evaluate the best model on the validation set
y_pred = knn.predict(X_test_scaled)

print("MSE")
print(mean_squared_error(y_test, y_pred))
print("==============")
print("RMSE")
print(mean_squared_error(y_test, y_pred, squared=False))
print("==============")
print("MAE")
print(mean_absolute_error(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("y actual")
plt.ylabel("y prediction")
plt.title("KNN Regression Test Error")
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.show()