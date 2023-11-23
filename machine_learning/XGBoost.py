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

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15],
    'min_child_weight': [5, 10, 15],
    'subsample': [0.8, 0.9, 1.0],
    'alpha': [0, 0.5, 1],
}

reg = XGBRegressor(alpha=0.5, learning_rate=0.2, n_estimators=500, max_depth=10, min_child_weight=15, subsample=0.9)

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
plt.title("XGBoost Test Error")
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