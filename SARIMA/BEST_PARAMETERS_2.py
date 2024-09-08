# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:59:41 2023

@author: catar
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from itertools import product

# Define parameters
start_date = '2012-01-01'
end_date = '2013-12-29'
k = 5

# Load and preprocess data
data=pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\PVproduction_modified_normalized_2.csv', parse_dates=['Time'],
                        usecols=["PV25", "Time"])
data.set_index('Time', inplace=True)
data_resampled = data.resample('1H').mean().shift(4)
data_resampled = data_resampled[data_resampled.index.date > data_resampled.index[0].date()]

X = data_resampled.iloc[:-24]
y = data_resampled.iloc[24:].values.ravel()

# Define parameter grids
p_values = [1, 2, 3]
d_values = [0]
q_values = [0]
P_values = [0]
D_values = [1, 2, 3]
Q_values = [0]
s_values = [24]

param_grid = product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values)

best_aic = np.inf
best_order = None
best_seasonal_order = None

for params in param_grid:
    p, d, q, P, D, Q, s = params
    try:
        model = SARIMAX(X, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False)
        model_fit = model.fit()
        aic = model_fit.aic
        if aic < best_aic:
            best_aic = aic
            best_order = (p, d, q)
            best_seasonal_order = (P, D, Q, s)
    except:
        continue

print("Best SARIMA Order:", best_order)
print("Best SARIMA Seasonal Order:", best_seasonal_order)

# Fit SARIMA model with best parameters and evaluate
sarima_model = SARIMAX(X, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False)
sarima_fit = sarima_model.fit()