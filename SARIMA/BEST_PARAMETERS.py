# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:05:44 2023

@author: catar
"""
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#factores escolhidos 
start_date = '2012-01-01'  # where trainning starts 
end_date = '2013-12-30'
k = 5

#### Save the real data for comparison in the plot
data_real = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\PVproduction_modified_normalized_2.csv', parse_dates=['Time'],
                        usecols=["PV25", "Time"])
data_real.set_index('Time', inplace=True)

# Resample the data to hourly intervals, using the mean of the previous four values
data_real_resampled = data_real.resample('1H').mean().shift(4)

# Drop the first day from the resampled data
data_real_resampled = data_real_resampled[data_real_resampled.index.date > data_real_resampled.index[0].date()]




#### Load data for prediction 
data =pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\PVproduction_modified_normalized_2.csv', parse_dates=['Time'],
                        usecols=["PV25", "Time"])
data.set_index('Time', inplace=True)

# Resample the data to hourly intervals, using the mean of the previous four values
data_resampled = data.resample('1H').mean().shift(4)

# Drop the first day from the resampled data
data_resampled = data_resampled[data_resampled.index.date > data_resampled.index[0].date()]

# Only use 3 months of data
data_resampled = data_resampled[(data_resampled.index >= start_date) & (data_resampled.index <= end_date)]

X = data_resampled.iloc[:-24]  # Historical data used for training, excluding the last 24 observations
y = data_resampled.iloc[24:]  # Values to predict, excluding the first 24 observations

# Reshape y to match the shape of X
y = y.values.ravel()


#### Parameter ranges to be test for the SARIMA model 
p = range(1, 4)
d = [0]
q = [0]
P = [0]
D = [1, 2, 3]
Q = [0]
s = [24]

#  k-fold cross-validation 
tscv = TimeSeriesSplit(n_splits=k) # split into train and test set 
mse_scores = []
rmse_scores = []
train_indices = []

for train_index, test_index in tscv.split(X):  # analising each fold , analyse the accurancy of the model 
    train_indices.append(train_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Find the best SARIMA parameters for each fold
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None

    for order in itertools.product(p, d, q):  # for each fold tries the differente comninations of Sarima parameters and selecets the model with the lowest Akaike information criterio
        for seasonal_order in itertools.product(P, D, Q, s):
            try:
                model = SARIMAX(X_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_seasonal_order = seasonal_order
            except:
                continue

    print("Best SARIMA Order:", best_order)
    print("Best SARIMA Order:",best_seasonal_order )






