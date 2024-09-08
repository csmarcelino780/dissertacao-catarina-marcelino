# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:46:35 2024

@author: catar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf  # Import the autocorrelation plot function
import holidays

# Load  data
start_date = '2012-02-01'  # where training starts 
end_date = '2013-06-29'  # where training ends 
k = 5

# clima file 
clima_file = r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Clustering de dados climaticos\mean_sheets_with_date.xlsx"

# consumption file 
data_df = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Normalizaçao Consumo\Normalized_Demand_Tese_teste.csv', parse_dates=['DateTime'], usecols=["MT_080_normalized", "DateTime"])

# solar file 
solar_data = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\PVproduction_modified_normalized_2.csv', parse_dates=['Time'],
                        usecols=["PV25", "Time"])
def load_real_data(data_df):
    data_real = data_df.copy()
    data_real.set_index('DateTime', inplace=True)
    return data_real

def resample_real_data(data_real):
    data_real_resampled = data_real.resample('1H').mean().shift(4)
    data_real_resampled = data_real_resampled[data_real_resampled.index.date > data_real_resampled.index[0].date()]
    return data_real_resampled

def load_and_preprocess_data(data_df):
    data = data_df.copy()
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['DayOfWeek'] = data['DateTime'].dt.dayofweek
    data['Day'] = data['DateTime'].dt.day
    data['Month'] = data['DateTime'].dt.month
    data['Hour'] = data['DateTime'].dt.hour
    data.set_index('DateTime', inplace=True)
    pt_holidays = holidays.Portugal()
    data['IsHoliday'] = data.index.map(lambda x: int(x in pt_holidays))
    return data

def resample_and_filter_data(data):
    data_resampled = data.resample('1H').mean().shift(4)
    data_resampled = data_resampled[data_resampled.index.date > data_resampled.index[0].date()]    
    return data_resampled

# Function to filter data for Summer and Winter months
def filter_seasonal_data(data, start_month, end_month):
    return data[(data.index.month >= start_month) & (data.index.month <= end_month)]

# Function to plot autocorrelation on given axes
def plot_autocorrelation(series, axes, lags=168, color='blue', label='', linewidth=2, markersize=5):
    plot_acf(series, ax=axes, lags=lags, alpha=None, color=color, vlines_kwargs={'linewidth': linewidth}, marker='o', markersize=markersize)
    axes.set_xlabel('Lag (hours)')
    axes.set_ylabel('Autocorrelation')
    axes.set_ylim(-0.5, 1)
    if label:
        axes.plot([], [], color=color, label=label)  # Add a dummy plot for label

# Load and preprocess data
data_real = load_real_data(data_df)
data_real_resampled = resample_real_data(data_real)
data = load_and_preprocess_data(data_df)
data_resampled = resample_and_filter_data(data)

# Use only the training set data
train_data = data_resampled[start_date:end_date]

# Filter data for Summer and Winter months
summer_data = filter_seasonal_data(data_resampled, 6, 8)

# Create a figure and axes for overlaying plots
fig, ax = plt.subplots(figsize=(15, 5))

# Plot autocorrelation for Consumption during the training period
plot_autocorrelation(train_data['MT_080_normalized'], ax, label='Training Period', linewidth=2, markersize=5)

# Plot autocorrelation for Consumption during Summer months
plot_autocorrelation(summer_data['MT_080_normalized'], ax, color='red', label='Summer Months', linewidth=2, markersize=5)

# Add title, legend, and show plot
ax.set_title('Autocorrelation - Consumption (Training Period and Summer Months)')
ax.legend()
plt.show()