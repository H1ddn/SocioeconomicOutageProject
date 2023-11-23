import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from scipy.stats import zscore

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

df = pd.read_csv('sorted_data/2016/data_2016.csv', converters={'FIPS' : str})

df['DATE'] = pd.to_datetime(df['DATE'])

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
X_train = df[feature_names].loc[(df['DATE'] > datetime.datetime(2016, 8, 27, 0, 0, 0)) & (df['DATE'] < datetime.datetime(2016, 9, 10, 0, 0, 0))]
y_train = df['next_day_avg'].loc[(df['DATE'] > datetime.datetime(2016, 8, 27, 0, 0, 0)) & (df['DATE'] < datetime.datetime(2016, 9, 10, 0, 0, 0))]
X_test = df[feature_names].loc[(df['DATE'] > datetime.datetime(2016, 10, 5, 0, 0, 0)) & (df['DATE'] < datetime.datetime(2016, 10, 10, 0, 0, 0))]
y_test = df['next_day_avg'].loc[(df['DATE'] > datetime.datetime(2016, 10, 5, 0, 0, 0)) & (df['DATE'] < datetime.datetime(2016, 10, 10, 0, 0, 0))]


params = {
    "n_estimators": 500,
    "max_features": 8,
    "max_depth": 15,
    "min_samples_split": 2,
    "random_state": 42,
    "verbose" : 1,
}

reg = RandomForestRegressor(**params)

reg.fit(X_train, y_train)

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
plt.title("Random Forest Test Error")
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

# obtain feature importance
feature_importance = reg.feature_importances_

# sort features according to importance
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0])

# plot feature importances
plt.barh(pos, feature_importance[sorted_idx], align="center")

plt.yticks(pos, np.array(feature_names)[sorted_idx])
plt.title("Feature Importance (MDI)")
plt.xlabel("Mean decrease in impurity");
plt.show()

df['prediction'] = reg.predict(X)

df.to_csv('sorted_data/visualize_2016.csv', index = False)
