# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:47:20 2024

@author: catar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf  # Import the autocorrelation plot function
import holidays
from statsmodels.tsa.stattools import acf

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
                        usecols=["PV50","PV25", "Time"])



def filter_seasonal_data(data, start_month, end_month):
    return data[(data.index.month >= start_month) & (data.index.month <= end_month)]

def manual_plot_autocorrelation(series, axes, lags=168, color='blue', label='', linewidth=0.5, markersize=1):
    # Calculate autocorrelation using statsmodels
    autocorr = acf(series, nlags=lags, fft=True)
    
    # Plot each lag as a separate line
    axes.vlines(range(len(autocorr)), [0], autocorr, color=color, linewidth=linewidth)
    axes.scatter(range(len(autocorr)), autocorr, color=color, s=markersize)  # Add scatter for visibility
    axes.axhline(y=0, color='black', linestyle='--', lw=0.5)  # Add a horizontal line at 0

    axes.set_xlabel('Lag (hours)')
    axes.set_ylabel('Autocorrelation')
    axes.set_ylim(-0.75, 1.1)  # Set y-axis range
    axes.set_title('Autocorrelation Plot')
    if label:
        axes.plot([], [], color=color, label=label)  # Add a dummy plot for label



####load 
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


# Load and preprocess data
data_real = load_real_data(data_df)
data_real_resampled = resample_real_data(data_real)
data = load_and_preprocess_data(data_df)
data_resampled = resample_and_filter_data(data)

# Use only the training set data
train_data = data_resampled[start_date:end_date]

# Filter data for Summer and Winter months
summer_data = filter_seasonal_data(data_resampled, 7, 8)
winter_data = filter_seasonal_data(data_resampled, 11, 12)

# # Create a figure and axes for overlaying plots
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot autocorrelation for Consumption during the training period
# manual_plot_autocorrelation(train_data['MT_080_normalized'], ax, color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)

# # Plot autocorrelation for Consumption during Summer months
# manual_plot_autocorrelation(summer_data['MT_080_normalized'], ax, color=(0.8, 0.2, 0.2), label='Summer Months', linewidth=0.5, markersize=1)

# # Add legend and show plot
# ax.legend()
# plt.show()

# # Create a figure and axes for overlaying plots
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot autocorrelation for Consumption during the training period
# manual_plot_autocorrelation(train_data['MT_080_normalized'], ax, color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)

# # Plot autocorrelation for Consumption during Winter months
# manual_plot_autocorrelation(winter_data['MT_080_normalized'], ax, color=(0.2, 0.8, 0.2), label='Winter Months', linewidth=0.5, markersize=1)

# # Add legend and show plot
# ax.legend()
# plt.show()

# Create a figure and axes for overlaying plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))  # Set ncols=2 for side-by-side plots

# Plot autocorrelation for Consumption during the training period and Summer months
manual_plot_autocorrelation(train_data['MT_080_normalized'], axes[0], color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)
manual_plot_autocorrelation(summer_data['MT_080_normalized'], axes[0], color=(0.8, 0.2, 0.2), label='Summer Months', linewidth=0.5, markersize=1)
axes[0].legend()
axes[0].set_title('A')
axes[0].set_ylim(-0.30, 1.1) 
# Plot autocorrelation for Consumption during the training period and Winter months
manual_plot_autocorrelation(train_data['MT_080_normalized'], axes[1], color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)
manual_plot_autocorrelation(winter_data['MT_080_normalized'], axes[1], color=(0.2, 0.8, 0.2), label='Winter Months', linewidth=0.5, markersize=1)
axes[1].legend()
axes[1].set_title('B')
axes[1].set_ylim(-0.30, 1.1) 

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()


#######solar 
def load_solar_real_data(solar_data):
    # Load and preprocess the real solar data
    solar_real = solar_data
    solar_real['Time'] = solar_real['Time'].apply(lambda x: x.replace(minute=0, second=0))
    solar_real['Time'] = pd.to_datetime(solar_real['Time'], format='%d/%m/%Y %H:%M')
    solar_real.set_index('Time', inplace=True)
    solar_real.index = pd.to_datetime(solar_real.index)
    solar_real = solar_real.dropna()
    # Resample the data to hourly intervals, using the mean of the previous four values
    solar_real_resampled = solar_real.resample('1H').mean().shift(4)
    return solar_real_resampled
def load_solar_data_for_prediction(solar_data):
    # Preprocess the solar data for prediction
    solar = solar_data.copy()
    solar['Time'] = solar['Time'].apply(lambda x: x.replace(minute=0, second=0))
    solar['Time'] = pd.to_datetime(solar['Time'], format='%d/%m/%Y %H:%M')
    solar.set_index('Time', inplace=True)
    solar.index = pd.to_datetime(solar.index)
    solar = solar.dropna()
    # Resample the data to hourly intervals, using the mean of the previous four values
    solar_resampled = solar.resample('1H').mean().shift(4)
    return solar_resampled

solar_resampled = load_solar_data_for_prediction(solar_data)
solar_real_resampled = load_solar_real_data(solar_data)

train_solar_data = solar_real_resampled[start_date:end_date]

# Filter data for Summer and Winter months
summer_solar_data = filter_seasonal_data(solar_real_resampled, 7, 8)
winter_solar_data = filter_seasonal_data(solar_real_resampled, 11, 12)

# # Create a figure and axes for overlaying plots
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot autocorrelation for Consumption during the training period
# manual_plot_autocorrelation(train_solar_data['PV25'], ax, color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)

# # Plot autocorrelation for Consumption during Summer months
# manual_plot_autocorrelation(summer_solar_data['PV25'], ax, color=(0.8, 0.2, 0.2), label='Summer Months', linewidth=0.5, markersize=1)

# # Add legend and show plot
# ax.legend()
# plt.show()

# # Create a figure and axes for overlaying plots
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot autocorrelation for Consumption during the training period
# manual_plot_autocorrelation(train_solar_data['PV25'], ax, color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)

# # Plot autocorrelation for Consumption during Winter months
# manual_plot_autocorrelation(winter_solar_data['PV25'], ax, color=(0.2, 0.8, 0.2), label='Winter Months', linewidth=0.5, markersize=1)

# # Add legend and show plot
# ax.legend()
# plt.show()

# Create a figure and axes for overlaying plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))  # Set ncols=2 for side-by-side plots

# Plot autocorrelation for Consumption during the training period and Summer months
manual_plot_autocorrelation(train_solar_data['PV50'], axes[0], color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)
manual_plot_autocorrelation(summer_solar_data['PV50'], axes[0], color=(0.8, 0.2, 0.2), label='Summer Months', linewidth=0.5, markersize=1)
axes[0].legend()
axes[0].set_title('A')
axes[0].set_ylim(-0.75, 1.1) 
# Plot autocorrelation for Consumption during the training period and Winter months
manual_plot_autocorrelation(train_solar_data['PV50'], axes[1], color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)
manual_plot_autocorrelation(winter_solar_data['PV50'], axes[1], color=(0.2, 0.8, 0.2), label='Winter Months', linewidth=0.5, markersize=1)
axes[1].legend()
axes[1].set_title('B')
axes[1].set_ylim(-0.75, 1.1) 
# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()


####net load 
# Calculate net load
net_load = data_resampled['MT_080_normalized'] - solar_resampled['PV50']

# Filter data for Summer and Winter months for net load
summer_net_load = filter_seasonal_data(net_load, 7, 8)
winter_net_load = filter_seasonal_data(net_load, 11, 12)

# Create a figure and axes for overlaying plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))

# Plot autocorrelation for Net Load during the training period and Summer months
manual_plot_autocorrelation(net_load[start_date:end_date], axes[0], color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)
manual_plot_autocorrelation(summer_net_load, axes[0], color=(0.8, 0.2, 0.2), label='Summer Months', linewidth=0.5, markersize=1)
axes[0].legend()
axes[0].set_title('A')
axes[0].set_ylim(-0.75, 1.1)

# Plot autocorrelation for Net Load during the training period and Winter months
manual_plot_autocorrelation(net_load[start_date:end_date], axes[1], color=(0.5, 0.5, 0.5), label='Training Period', linewidth=0.5, markersize=1)
manual_plot_autocorrelation(winter_net_load, axes[1], color=(0.2, 0.8, 0.2), label='Winter Months', linewidth=0.5, markersize=1)
axes[1].legend()
axes[1].set_title('B')
axes[1].set_ylim(-0.75, 1.1)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()








# Calculate net load
net_load = data_resampled['MT_080_normalized'] - solar_resampled['PV50']

# Filter data for Summer and Winter months for net load
summer_net_load = filter_seasonal_data(net_load,7,8)
winter_net_load = filter_seasonal_data(net_load,11,12)



# Create a figure and axes for the net load daily profile
fig, ax = plt.subplots(figsize=(8, 5))

# Plot Net Load daily profile for Summer months
summer_net_load_daily_profile = summer_net_load.groupby(summer_net_load.index.hour).mean()
ax.plot(summer_net_load_daily_profile.index, summer_net_load_daily_profile, color='orange', label='Summer Months')

# Plot Net Load daily profile for Winter months
winter_net_load_daily_profile = winter_net_load.groupby(winter_net_load.index.hour).mean()
ax.plot(winter_net_load_daily_profile.index, winter_net_load_daily_profile, color='blue', label='Winter Months')

ax.set_title('Net Load Daily Profile')
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Net Load')
ax.set_ylim(-5, 5)  # Set consistent y-axis limits to (-5, 5)
ax.legend()

# Show the plot
plt.show()


# Create a figure and axes for the load daily profile
fig, axes_load = plt.subplots(figsize=(8, 5))

# Plot Load daily profile for Summer months
summer_data_daily_profile = summer_data.groupby(summer_data.index.hour)['MT_080_normalized'].mean()
axes_load.plot(summer_data_daily_profile.index, summer_data_daily_profile, color='orange', label='Summer Load')

# Plot Load daily profile for Winter months
winter_data_daily_profile = winter_data.groupby(winter_data.index.hour)['MT_080_normalized'].mean()
axes_load.plot(winter_data_daily_profile.index, winter_data_daily_profile, color='blue', label='Winter Load')

axes_load.set_title('Load Daily Profile')
axes_load.set_xlabel('Hour of the Day')
axes_load.set_ylabel('Normalized Load')
axes_load.set_ylim(-2, 5)  # Set consistent y-axis limits to (-2, 5)
axes_load.legend()

# Show the load plot
plt.show()



# Create a figure and axes for the solar daily profile
fig, axes_solar = plt.subplots(figsize=(8, 5))

# Plot Solar daily profile for Summer months
summer_solar_data_daily_profile = summer_solar_data.groupby(summer_solar_data.index.hour)['PV50'].mean()
axes_solar.plot(summer_solar_data_daily_profile.index, summer_solar_data_daily_profile, color='orange', label='Summer Solar')

# Plot Solar daily profile for Winter months
winter_solar_data_daily_profile = winter_solar_data.groupby(winter_solar_data.index.hour)['PV50'].mean()
axes_solar.plot(winter_solar_data_daily_profile.index, winter_solar_data_daily_profile, color='blue', label='Winter Solar')

axes_solar.set_title('Solar Daily Profile')
axes_solar.set_xlabel('Hour of the Day')
axes_solar.set_ylabel('Normalized Solar Production')
axes_solar.set_ylim(-2, 5)  # Set consistent y-axis limits to (-2, 5)
axes_solar.legend()

# Show the solar plot
plt.show()

