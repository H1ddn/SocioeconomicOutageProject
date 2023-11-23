import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from scipy.stats import zscore
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

df = pd.read_csv('sorted_data/2016/data_2016.csv', converters={'FIPS' : str})

print(df['next_day_avg'].std())
print(df['next_day_avg'].mean())
print(df.shape)

feature_names = ['outage_max', 'outage_max_3d', 'outage_avg', 'outage_avg_3d', 'outage_duration', 'outage_duration_3d',
        'Pov_pct',   
       'Med_inc', 'Pop_est', 'Unemp_rate','Soc_vuln', 'Comm_res', 'Tornado_score',
       'Hurricane_score', 'River_flood_score', 'Coast_flood_score', 'Tair_f',  
       'Qair_f', 'Psurf_f', 'Wind_f', 'Rainf_f', 'FloodedFrac', 'Qs', 'Qsb',   
       'Streamflow', 'Res_elec', 'Res_customers', 'Cml_elec', 'Cml_customers',
       'Wind_f_3d','Rainf_f_3d','FloodedFrac_3d','Qs_3d','Qsb_3d','Streamflow_3d']

X = df[feature_names]

y = df['next_day_avg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Range of alpha values to test
    'fit_intercept': [True, False],  # Whether to fit an intercept
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']  # Solver options
}

reg = Ridge(alpha=0.001, fit_intercept=True, solver='auto')

reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

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
plt.title("Linear Regression Test Error")
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

