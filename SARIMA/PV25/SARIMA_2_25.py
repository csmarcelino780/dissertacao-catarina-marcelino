# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:11:37 2023

@author: catar
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import holidays
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from statsmodels.tools.eval_measures import aic
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Load  data
start_date = '2012-02-01'  # where trainning starts 
end_date = '2013-06-29'  # where traning ends 
k = 15

# consumption file 
data_df = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Normalizaçao Consumo\Normalized_Demand_Tese_teste.csv', parse_dates=['DateTime'], usecols=["MT_080_normalized", "DateTime"])
data_df.set_index(pd.to_datetime(data_df['DateTime']), inplace=True)

# solar file 
solar_data = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\PVproduction_modified_normalized_2.csv', parse_dates=['Time'],
                        usecols=["PV25", "Time"])

###################################################load ###############################
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

# treat the data 
data_real = load_real_data(data_df)
data_real_resampled = resample_real_data(data_real)
data = load_and_preprocess_data(data_df)
data_resampled = resample_and_filter_data(data)

X = data_resampled.iloc[:-24].dropna() # exclud the last 24 rows + dropping rows with NaN values
valid_indices = X.index # Store the indices of the non-NaN rows in X

y = data_resampled['MT_080_normalized'].iloc[24:].values #exclud the first 24 values
valid_positions = [data_resampled.index.get_loc(idx) for idx in valid_indices] #list of positions in data_resampled for the valid_indices
y = y[valid_positions] #keeping only the values corresponding to non-NaN rows in X

# Best SARIMA parameters (you should set these based on your pre-determined best parameters)
best_order = (3,0,0)
best_seasonal_order = (0,1,0,24)
sarima_model = SARIMAX(endog=data_df['MT_080_normalized'], order=best_order, seasonal_order=best_seasonal_order, exog=None)


# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=k)

# Store evaluation metrics for SARIMA
sarima_mse_scores = []
sarima_rmse_scores = []

sarima_mse_scores = []
sarima_rmse_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply the date range filter to both X_train and y_train
    common_dates = (X_train.index >= start_date) & (X_train.index <= end_date)
    X_train = X_train[common_dates]
    y_train = y_train[common_dates]

    # Initialize the SARIMA model with hyperparameters
    sarima_model = SARIMAX(endog=y_train, order=best_order, seasonal_order=best_seasonal_order, exog=None)
    
    # Fit the SARIMA model on the training data
    sarima_fit = sarima_model.fit()

    # Make forecasts on the test data
    forecast = sarima_fit.get_forecast(steps=len(X_test))
    forecasted_load = forecast.predicted_mean

    # Calculate MSE and RMSE for this fold
    sarima_mse = mean_squared_error(y_test, forecasted_load)
    sarima_rmse = np.sqrt(sarima_mse)

    sarima_mse_scores.append(sarima_mse)
    sarima_rmse_scores.append(sarima_rmse)

# Calculate average MSE and RMSE
avg_sarima_mse = np.mean(sarima_mse_scores)
avg_sarima_rmse = np.mean(sarima_rmse_scores)



print(f"Average SARIMA MSE: {avg_sarima_mse}")
print(f"Average SARIMA RMSE: {avg_sarima_rmse}")

# Months to forecast
months = [7, 8]  # July and August
days_in_month = [31, 31]  # Number of days in July and August, adjust as needed

monthly_rmse = {month: [] for month in months}
monthly_mse = {month: [] for month in months}

# Convert end_date to a Timestamp
end_date = pd.to_datetime(end_date)

hourly_rmse = []
hourly_mse = []

forecasted_loads = []

for month in [7, 8]:
    for day in range(1, days_in_month[month - 7] + 1):
        # Update the SARIMA model using the data up to the current prediction date
        current_data = X_train.loc[X_train.index <= pd.to_datetime(f"2013-{month:02d}-{day:02d}")]
        endog = current_data['MT_080_normalized']
        sarima_model = SARIMAX(endog=endog, order=best_order, seasonal_order=best_seasonal_order, exog=None)
        
        sarima_fit = sarima_model.fit()
        
        for hour in range(24):
            prediction_datetime = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")
            
            # Make the hour-ahead forecast using the updated SARIMA model
            forecasted_load = sarima_fit.predict(start=len(current_data) + hour, end=len(current_data) + hour)
            
            # Append the forecasted load and timestamp to the results
            forecasted_loads.append((prediction_datetime, forecasted_load[0]))

            # Extract the observed values for the same time from data_real_resampled
            observed_values = data_real_resampled.loc[prediction_datetime].values
            
                # Calculate RMSE for the current hour
            rmse = np.sqrt(mean_squared_error(observed_values, forecasted_load))
            hourly_rmse.append(rmse)
                
                # Calculate MSE for the current hour
            mse = mean_squared_error(observed_values, forecasted_load)
            hourly_mse.append(mse)

# Forecasts to a DataFrame
forecasted_loads_df = pd.DataFrame(forecasted_loads, columns=["DateTime", "Forecasted_Load"])
forecasted_loads_df.set_index("DateTime", inplace=True)

# Forecasts to a DataFrame
forecasted_loads_df = pd.DataFrame(forecasted_loads, columns=["DateTime", "Forecasted_Load"])
forecasted_loads_df.set_index("DateTime", inplace=True)
                
# Calculate monthly RMSE by taking the average for each month
july_rmse = np.mean(hourly_rmse[:31 * 24])  # RMSE for July
august_rmse = np.mean(hourly_rmse[31 * 24:])  # RMSE for August
july_mse = np.mean(hourly_mse[:31 * 24])  # MSE for July
august_mse = np.mean(hourly_mse[31 * 24:])  # MSE for August

print("Average RMSE for July:", july_rmse)
print("Average RMSE for August:", august_rmse)
# ############################################ sOLAR ################################################
# solar file 
solar_data = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\PVproduction_modified_normalized_2.csv', parse_dates=['Time'],
                        usecols=["PV25", "Time"])

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

# treat the data 
solar_resampled = load_solar_data_for_prediction(solar_data)
solar_real_resampled = load_solar_real_data(solar_data)

X_solar = solar_resampled.iloc[:-24].dropna() # exclud the last 24 rows + dropping rows with NaN values
valid_indices_solar = X_solar.index # Store the indices of the non-NaN rows in X
y_solar = solar_resampled['PV25'].iloc[24:].values #exclud the first 24 values
valid_positions_solar = [solar_resampled.index.get_loc(idx) for idx in valid_indices_solar] #list of positions in data_resampled for the valid_indices
y_solar = y_solar[valid_positions_solar] #keeping only the values corresponding to non-NaN rows in X

# Initialize SARIMA model with the best parameters
sarima_model_solar = SARIMAX(endog=solar_data['PV25'], order=best_order, seasonal_order=best_seasonal_order)

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=k)

mse_scores_solar = []
rmse_scores_solar = []

# Train and evaluate the SARIMA model using cross-validation for solar data
for train_index_solar, test_index_solar in tscv.split(X_solar):
    
    X_train_solar, X_test_solar = X_solar.iloc[train_index_solar], X_solar.iloc[test_index_solar]
    y_train_solar, y_test_solar = y_solar[train_index_solar], y_solar[test_index_solar]

    # Apply the date range filter to both X_train_solar and y_train_solar
    common_dates_solar = (X_train_solar.index >= start_date) & (X_train_solar.index <= end_date)
    X_train_solar = X_train_solar[common_dates_solar]
    y_train_solar = y_train_solar[common_dates_solar]
    
    # Initialize SARIMA model with the best parameters
    sarima_model_solar = SARIMAX(endog=y_train_solar, order=best_order, seasonal_order=best_seasonal_order)
    
    # Fit the SARIMA model
    sarima_solar_fit = sarima_model_solar.fit(disp=False)

    # Forecast using SARIMA
    forecasted_solar = sarima_solar_fit.forecast(steps=24, exog=None)

    # Evaluate the SARIMA model for solar data
    mse_solar = mean_squared_error(y_test_solar[-24:], forecasted_solar)
    mse_scores_solar.append(mse_solar)

    rmse_solar = np.sqrt(mse_solar)
    rmse_scores_solar.append(rmse_solar)

# Calculate average MSE and RMSE for solar data
avg_mse_solar = np.mean(mse_scores_solar)
avg_rmse_solar = np.mean(rmse_scores_solar)

# Output the evaluation metrics for solar data
print(f"Average MSE for Solar: {avg_mse_solar}")
print(f"Average RMSE for Solar: {avg_rmse_solar}")

# Calculate the maximum value for the same month of the previous year
for month in [7, 8]:  # July and August
    last_year_month = month - 1  # Month of the previous year
    if last_year_month == 0:
        last_year_month = 12  # December of the previous year

    max_value_last_year = solar_real_resampled[solar_real_resampled.index.month == last_year_month]['PV25'].max()

forecasted_solar = []

#  Months to forcast 
months_solar = [7, 8]  # July and August
days_in_month_solar = [31, 31]  # Number of days in July and August, adjust as needed

monthly_rmse_solar = {month: [] for month in months_solar}
monthly_mse_solar = {month: [] for month in months_solar}

# Convert end_date_solar to a Timestamp
end_date_solar = pd.to_datetime(end_date)

hourly_rmse_solar = []
hourly_mse_solar = []  

forecasted_solar = []  # Initialize the list to store forecasts

for month in [7, 8]:
    for day in range(1, days_in_month[month - 7] + 1):
        # Update the SARIMA model using the data up to the current prediction date
        current_data = X_train_solar.loc[X_train_solar.index <= pd.to_datetime(f"2013-{month:02d}-{day:02d}")]
        endog = current_data['PV25']
        sarima_model_solar = SARIMAX(endog=endog, order=best_order, seasonal_order=best_seasonal_order, exog=None)
        
        sarima_fit = sarima_model.fit()
        
        for hour in range(24):
            prediction_datetime_solar = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")
            
            # Make the hour-ahead forecast using the updated SARIMA model
            forecasted_solar_values = sarima_solar_fit.predict(start=len(current_data) + hour, end=len(current_data) + hour)
            
                # Adjust forecasted values to meet min and max constraints
            forecasted_solar_values = np.maximum(0, forecasted_solar_values)  # Apply min constraint
            forecasted_solar_values = np.minimum(max_value_last_year, forecasted_solar_values)  # Apply max constraint

            forecasted_solar.append((prediction_datetime_solar, forecasted_solar_values[0]))

                # Extract the observed values for the same time from solar_real_resampled
            observed_values_solar = solar_real_resampled.loc[prediction_datetime_solar].values
                
                # Calculate RMSE for the current hour for solar data
            rmse_solar = np.sqrt(mean_squared_error(observed_values_solar, forecasted_solar_values))
            hourly_rmse_solar.append(rmse_solar)
                
                # Calculate MSE for the current hour for solar data
            mse_solar = mean_squared_error(observed_values_solar, forecasted_solar_values)
            hourly_mse_solar.append(mse_solar)

# Convert forecasts to a DataFrame
forecast_solar_df = pd.DataFrame(forecasted_solar, columns=["DateTime", "Forecasted_Solar"])
forecast_solar_df.set_index("DateTime", inplace=True)


# Calculate monthly RMSE for solar data by taking the average for each month
july_rmse_solar = np.mean(hourly_rmse_solar[:31 * 24])  # RMSE for July for solar data
august_rmse_solar = np.mean(hourly_rmse_solar[31 * 24:])  # RMSE for August for solar data

# Calculate monthly MSE for solar data by taking the average for each month
july_mse_solar = np.mean(hourly_mse_solar[:31 * 24])  # MSE for July for solar data
august_mse_solar = np.mean(hourly_mse_solar[31 * 24:])  # MSE for August for solar data

print("Average RMSE for July (Solar):", july_rmse_solar)
print("Average RMSE for August (Solar):", august_rmse_solar)

##################### net load ##########################################################
# Filter the real load and solar data for July and August
real_load_prediction_days = data_real_resampled.loc[data_real_resampled.index.month.isin([7, 8])]['MT_080_normalized']
real_solar_prediction_days = solar_real_resampled.loc[solar_real_resampled.index.month.isin([7, 8])]['PV25']  # Assuming solar data is used for solar_real

# Calculate the real net consumption for the selected days
net_consumption = real_load_prediction_days - real_solar_prediction_days

# Set negative values to 0
#net_consumption[net_consumption < 0] = 0

# Convert net_consumption to a list
net_consumption_values = net_consumption.tolist()


#PV25
mean_net_consumption = 11.335933
std_net_consumption = 6.409994163

#PV50 
# mean_net_consumption = 9.384514695
# std_net_consumption = 7.714198377

############################## method 1 ( forecasted load - forecasted solar)#############################
months_to_calculate = [7, 8]  # July and August
days_in_month = [31, 31]  

# Store hourly evaluation metrics for net consumption
hourly_mse_net_consumption = []
hourly_rmse_net_consumption = []
hourly_mae_net_consumption = []  # Added for MAE
hourly_mape_net_consumption = []
net_consumption_per_day = []
mse_per_day = []
rmse_per_day = []
net_forecast_values = []  # Added to store forecasted values
real_net_consumption_values = [] 

for month in [7, 8]:  # July and August for net consumption data
    for day in range(1, 32):
        for hour in range(24):  
            prediction_datetime_solar = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")
        
            #  prediction day should be excluded (e.g., before your data starts)
            if prediction_datetime_solar >= data_real_resampled.index[0]:
                # Calculate the real net consumption for the current day and hour
                real_load = real_load_prediction_days.loc[prediction_datetime_solar]
                real_solar = real_solar_prediction_days.loc[prediction_datetime_solar]
                real_net_consumption = real_load - real_solar

                # Append the calculated values to the respective lists
                net_consumption_per_day.append(real_net_consumption)
                real_net_consumption_values.append(real_net_consumption)
                
                # Calculate the net load forecast for the current day and hour
                net_forecast = forecasted_loads_df.loc[prediction_datetime_solar]["Forecasted_Load"] - forecast_solar_df.loc[prediction_datetime_solar]["Forecasted_Solar"]
                net_forecast_values.append(net_forecast)  # Store forecasted values

                # Calculate the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) for the current hour
                mse = mean_squared_error([real_net_consumption], [net_forecast])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error([real_net_consumption], [net_forecast])
                mape = mean_absolute_percentage_error([real_net_consumption], [net_forecast])

                # Append the calculated MSE, RMSE, and MAE to the respective lists
                hourly_mse_net_consumption.append(mse)
                hourly_rmse_net_consumption.append(rmse)
                hourly_mae_net_consumption.append(mae)
                hourly_mape_net_consumption.append(mape)


# Denormalize the errors for overall
# Denormalize the errors for overall
overall_mse_nl1 = np.array(hourly_mse_net_consumption) * std_net_consumption + mean_net_consumption
overall_rmse_nl1 = np.array(hourly_rmse_net_consumption) * std_net_consumption + mean_net_consumption
overall_mae_nl1 = np.array(hourly_mae_net_consumption) * std_net_consumption + mean_net_consumption
overall_mape_nl1 = np.array(hourly_mape_net_consumption)* std_net_consumption / mean_net_consumption 

# Calculate overall averages
overall_average_mse_nl1 = np.mean(overall_mse_nl1)
overall_average_rmse_nl1 = np.mean(overall_rmse_nl1)
overall_average_mae_nl1 = np.mean(overall_mae_nl1)
overall_average_mape_nl1 = np.mean(overall_mape_nl1)

# Print or use the overall averages as needed
print("Overall Average MSE: ", overall_average_mse_nl1)
print("Overall Average RMSE: ", overall_average_rmse_nl1)
print("Overall Average MAE: ", overall_average_mae_nl1)
print("Overall Average MAPE: ", overall_average_mape_nl1)


# Denormalize the errors for July
hourly_mse_july_nl1 = np.array(hourly_mse_net_consumption[:31*24]) * std_net_consumption + mean_net_consumption
hourly_rmse_july_nl1 = np.array(hourly_rmse_net_consumption[:31*24]) * std_net_consumption + mean_net_consumption
hourly_mae_july_nl1 = np.array(hourly_mae_net_consumption[:31*24]) * std_net_consumption + mean_net_consumption
hourly_mape_july_nl1 = np.array(hourly_mape_net_consumption[:31*24])* std_net_consumption / mean_net_consumption 

# Denormalize the errors for August
hourly_mse_august_nl1 = np.array(hourly_mse_net_consumption[31*24:]) * std_net_consumption + mean_net_consumption
hourly_rmse_august_nl1 = np.array(hourly_rmse_net_consumption[31*24:]) * std_net_consumption + mean_net_consumption
hourly_mae_august_nl1 = np.array(hourly_mae_net_consumption[31*24:]) * std_net_consumption + mean_net_consumption
hourly_mape_august_nl1 = np.array(hourly_mape_net_consumption[31*24:])* std_net_consumption / mean_net_consumption 


# Reshape the arrays for July into (24, 31) arrays
hourly_rmse_july_nl1_reshaped = hourly_rmse_july_nl1.reshape((24, 31), order='F')
hourly_mse_july_nl1_reshaped = hourly_mse_july_nl1.reshape((24, 31), order='F')
hourly_mae_july_nl1_reshaped = hourly_mae_july_nl1.reshape((24, 31), order='F')
hourly_mape_july_nl1_reshaped = hourly_mape_july_nl1.reshape((24, 31), order='F')

# Calculate the mean for each hour across all Mondays, Tuesdays, etc.
mean_rmse_per_hour_across_days = np.mean(hourly_rmse_july_nl1_reshaped, axis=1)
mean_mse_per_hour_across_days = np.mean(hourly_mse_july_nl1_reshaped, axis=1)
mean_mae_per_hour_across_days = np.mean(hourly_mae_july_nl1_reshaped, axis=1)
mean_mape_per_hour_across_days = np.mean(hourly_mape_july_nl1_reshaped, axis=1)


# Create DataFrames for each metric
rmse_df_july = pd.DataFrame({'Hour': range(24),
                        'Mean_RMSE_Monday': hourly_rmse_july_nl1_reshaped[:, 0],
                        'Mean_RMSE_Tuesday': hourly_rmse_july_nl1_reshaped[:, 1],
                        'Mean_RMSE_Wednesday': hourly_rmse_july_nl1_reshaped[:, 2],
                        'Mean_RMSE_Thursday': hourly_rmse_july_nl1_reshaped[:, 3],
                        'Mean_RMSE_Friday': hourly_rmse_july_nl1_reshaped[:, 4],
                        'Mean_RMSE_Saturday': hourly_rmse_july_nl1_reshaped[:, 5],
                        'Mean_RMSE_Sunday': hourly_rmse_july_nl1_reshaped[:, 6]})

mse_df_july = pd.DataFrame({'Hour': range(24),
                       'Mean_MSE_Monday': hourly_mse_july_nl1_reshaped[:, 0],
                       'Mean_MSE_Tuesday': hourly_mse_july_nl1_reshaped[:, 1],
                       'Mean_MSE_Wednesday': hourly_mse_july_nl1_reshaped[:, 2],
                       'Mean_MSE_Thursday': hourly_mse_july_nl1_reshaped[:, 3],
                       'Mean_MSE_Friday': hourly_mse_july_nl1_reshaped[:, 4],
                       'Mean_MSE_Saturday': hourly_mse_july_nl1_reshaped[:, 5],
                       'Mean_MSE_Sunday': hourly_mse_july_nl1_reshaped[:, 6]})

mae_df_july = pd.DataFrame({'Hour': range(24),
                       'Mean_MAE_Monday': hourly_mae_july_nl1_reshaped[:, 0],
                       'Mean_MAE_Tuesday': hourly_mae_july_nl1_reshaped[:, 1],
                       'Mean_MAE_Wednesday': hourly_mae_july_nl1_reshaped[:, 2],
                       'Mean_MAE_Thursday': hourly_mae_july_nl1_reshaped[:, 3],
                       'Mean_MAE_Friday': hourly_mae_july_nl1_reshaped[:, 4],
                       'Mean_MAE_Saturday': hourly_mae_july_nl1_reshaped[:, 5],
                       'Mean_MAE_Sunday': hourly_mae_july_nl1_reshaped[:, 6]})

mape_df_july = pd.DataFrame({'Hour': range(24),
                              'Mean_MAPE_Monday': hourly_mape_july_nl1_reshaped[:, 0],
                              'Mean_MAPE_Tuesday': hourly_mape_july_nl1_reshaped[:, 1],
                              'Mean_MAPE_Wednesday': hourly_mape_july_nl1_reshaped[:, 2],
                              'Mean_MAPE_Thursday': hourly_mape_july_nl1_reshaped[:, 3],
                              'Mean_MAPE_Friday': hourly_mape_july_nl1_reshaped[:, 4],
                              'Mean_MAPE_Saturday': hourly_mape_july_nl1_reshaped[:, 5],
                              'Mean_MAPE_Sunday': hourly_mape_july_nl1_reshaped[:, 6]})

hourly_rmse_august_nl1_reshaped = hourly_rmse_august_nl1.reshape((24, 31), order='F')
hourly_mse_august_nl1_reshaped = hourly_mse_august_nl1.reshape((24, 31), order='F')
hourly_mae_august_nl1_reshaped = hourly_mae_august_nl1.reshape((24, 31), order='F')
hourly_mape_august_nl1_reshaped = hourly_mape_august_nl1.reshape((24, 31), order='F')

# Reshape the arrays for August into (24, 31) arrays
# Calculate the mean for each hour across all Mondays, Tuesdays, etc.
mean_rmse_per_hour_across_days_august = np.mean(hourly_rmse_august_nl1_reshaped, axis=1)
mean_mse_per_hour_across_days_august = np.mean(hourly_mse_august_nl1_reshaped, axis=1)
mean_mae_per_hour_across_days_august = np.mean(hourly_mae_august_nl1_reshaped, axis=1)
mean_mape_per_hour_across_days_august = np.mean(hourly_mape_august_nl1_reshaped, axis=1)
# Create DataFrames for each metric
rmse_df_august = pd.DataFrame({'Hour': range(24),
                               'Mean_RMSE_Monday': hourly_rmse_august_nl1_reshaped[:, 0],
                               'Mean_RMSE_Tuesday': hourly_rmse_august_nl1_reshaped[:, 1],
                               'Mean_RMSE_Wednesday': hourly_rmse_august_nl1_reshaped[:, 2],
                               'Mean_RMSE_Thursday': hourly_rmse_august_nl1_reshaped[:, 3],
                               'Mean_RMSE_Friday': hourly_rmse_august_nl1_reshaped[:, 4],
                               'Mean_RMSE_Saturday': hourly_rmse_august_nl1_reshaped[:, 5],
                               'Mean_RMSE_Sunday': hourly_rmse_august_nl1_reshaped[:, 6]})

mse_df_august = pd.DataFrame({'Hour': range(24),
                              'Mean_MSE_Monday': hourly_mse_august_nl1_reshaped[:, 0],
                              'Mean_MSE_Tuesday': hourly_mse_august_nl1_reshaped[:, 1],
                              'Mean_MSE_Wednesday': hourly_mse_august_nl1_reshaped[:, 2],
                              'Mean_MSE_Thursday': hourly_mse_august_nl1_reshaped[:, 3],
                              'Mean_MSE_Friday': hourly_mse_august_nl1_reshaped[:, 4],
                              'Mean_MSE_Saturday': hourly_mse_august_nl1_reshaped[:, 5],
                              'Mean_MSE_Sunday': hourly_mse_august_nl1_reshaped[:, 6]})

mae_df_august = pd.DataFrame({'Hour': range(24),
                              'Mean_MAE_Monday': hourly_mae_august_nl1_reshaped[:, 0],
                              'Mean_MAE_Tuesday': hourly_mae_august_nl1_reshaped[:, 1],
                              'Mean_MAE_Wednesday': hourly_mae_august_nl1_reshaped[:, 2],
                              'Mean_MAE_Thursday': hourly_mae_august_nl1_reshaped[:, 3],
                              'Mean_MAE_Friday': hourly_mae_august_nl1_reshaped[:, 4],
                              'Mean_MAE_Saturday': hourly_mae_august_nl1_reshaped[:, 5],
                              'Mean_MAE_Sunday': hourly_mae_august_nl1_reshaped[:, 6]})

# Create a DataFrame for MAPE in August
mape_df_august = pd.DataFrame({'Hour': range(24),
                               'Mean_MAPE_Monday': hourly_mape_august_nl1_reshaped[:, 0],
                               'Mean_MAPE_Tuesday': hourly_mape_august_nl1_reshaped[:, 1],
                               'Mean_MAPE_Wednesday': hourly_mape_august_nl1_reshaped[:, 2],
                               'Mean_MAPE_Thursday': hourly_mape_august_nl1_reshaped[:, 3],
                               'Mean_MAPE_Friday': hourly_mape_august_nl1_reshaped[:, 4],
                               'Mean_MAPE_Saturday': hourly_mape_august_nl1_reshaped[:, 5],
                               'Mean_MAPE_Sunday': hourly_mape_august_nl1_reshaped[:, 6]})

# Calculate the average MSE, RMSE, and MAE for July and August
average_mse_july_nl1 = np.mean(hourly_mse_july_nl1)
average_rmse_july_nl1 = np.mean(hourly_rmse_july_nl1)
average_mae_july_nl1 = np.mean(hourly_mae_july_nl1)
average_mape_july_nl1 = np.mean(hourly_mape_july_nl1)
average_mse_august_nl1 = np.mean(hourly_mse_august_nl1)
average_rmse_august_nl1 = np.mean(hourly_rmse_august_nl1)
average_mae_august_nl1 = np.mean(hourly_mae_august_nl1)
average_mape_august_nl1 = np.mean(hourly_mape_august_nl1)


############################## Method 2 (SARIMA forecasts) #############################
# Load and preprocess net load and solar data
net_load = data_resampled['MT_080_normalized'] - solar_resampled['PV25']

if isinstance(net_load, pd.Series):
    net_load = pd.DataFrame(net_load, columns=['Net_Load'])

# Resampling (if needed)
net_load = net_load.resample('H').mean()

# Drop NaN values (if needed)
net_load.dropna(inplace=True)

X_net = net_load.iloc[:-24].dropna()  # Exclude the last 24 rows + dropping rows with NaN values
y_net = net_load.iloc[24:].values

# Initialize lists to store evaluation metrics
mse_scores_net = []
rmse_scores_net = []

# SARIMA model for net load
sarima_net = SARIMAX(endog=y_net, order=best_order, seasonal_order=best_seasonal_order)

# TimeSeriesSplit for cross-validation
tscv_net = TimeSeriesSplit(n_splits=k)

# Train and evaluate the SARIMA model using cross-validation
for train_index, test_index in tscv_net.split(X_net):
    X_train_net, X_test_net = X_net.iloc[train_index], X_net.iloc[test_index]
    y_train_net, y_test_net = y_net[train_index], y_net[test_index]

    # Fit the SARIMA model
    sarima_fit = sarima_net.fit(disp=False)

    # Forecast using SARIMA
    forecasted_load_net = sarima_fit.forecast(steps=len(y_test_net))

    # Evaluate the model
    mse_net = mean_squared_error(y_test_net, forecasted_load_net)
    mse_scores_net.append(mse_net)

    rmse_net = np.sqrt(mse_net)
    rmse_scores_net.append(rmse_net)

# Calculate average MSE and RMSE for the net load
avg_mse_net = np.mean(mse_scores_net)
avg_rmse_net = np.mean(rmse_scores_net)

# Output the evaluation metrics for the net load
print(f"Average MSE for Net Load: {avg_mse_net}")
print(f"Average RMSE for Net Load: {avg_rmse_net}")

forecasted_loads_net = []

# Define the months and days
months = [7, 8]  # July and August
days_in_month = [31, 31]  # Number of days in July and August, adjust as needed

monthly_rmse = {month: [] for month in months}
monthly_mse = {month: [] for month in months}

# Convert end_date to a Timestamp
end_date = pd.to_datetime(end_date)

# Store hourly evaluation metrics for net load
hourly_mse_net = []
hourly_rmse_net = []
hourly_mae_net = []  # Added for MAE
hourly_mape_net = [] 
# Initialize lists to store RMSE, MSE, and MAE per weekday
rmse_per_weekday_net = [0] * 7
mse_per_weekday_net = [0] * 7
mae_per_weekday_net = [0] * 7
count_per_weekday_net = [0] * 7

for month in [7, 8]:
    for day in range(1, days_in_month[month - 7] + 1):
        # Update the SARIMA model using the data up to the current prediction date
        current_data_net = net_load.loc[net_load.index <= pd.to_datetime(f"2013-{month:02d}-{day:02d}")]
        endog_net = current_data_net  # Use net_load as endog_net

        sarima_model_net = SARIMAX(endog=endog_net, order=best_order, seasonal_order=best_seasonal_order, exog=None)

        sarima_fit_net = sarima_model_net.fit()

        for hour in range(24):
            prediction_datetime_net = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")

            # Make the hour-ahead forecast using the updated SARIMA model
            forecasted_load_net = sarima_fit_net.predict(start=len(current_data_net) + hour, end=len(current_data_net) + hour)

            # Append the forecasted load to the results
            forecasted_loads_net.append(forecasted_load_net[0])

            # Extract the observed values for the same time from net_load
            observed_values_net = net_load.loc[prediction_datetime_net].values

            # Calculate RMSE, MSE, and MAE for the current hour for net load
            rmse_net = np.sqrt(mean_squared_error(observed_values_net, forecasted_load_net))
            mse_net = mean_squared_error(observed_values_net, forecasted_load_net)
            mae_net = mean_absolute_error(observed_values_net, forecasted_load_net)
            mape_net = mean_absolute_percentage_error(observed_values_net, forecasted_load_net)
            hourly_rmse_net.append(rmse_net)
            hourly_mse_net.append(mse_net)
            hourly_mae_net.append(mae_net)
            hourly_mape_net.append(mape_net)
                        
# Denormalize the errors for overall
overall_mse_net = np.array(hourly_mse_net) * std_net_consumption + mean_net_consumption
overall_rmse_net = np.array(hourly_rmse_net) * std_net_consumption + mean_net_consumption
overall_mae_net = np.array(hourly_mae_net) * std_net_consumption + mean_net_consumption
overall_mape_net = np.array(hourly_mape_net)* std_net_consumption / mean_net_consumption 

# Calculate overall averages
overall_average_mse_net = np.mean(overall_mse_net)
overall_average_rmse_net = np.mean(overall_rmse_net)
overall_average_mae_net = np.mean(overall_mae_net)
overall_average_mape_net = np.mean(overall_mape_net)

# Print or use the overall averages as needed
print("Overall Average MSE: ", overall_average_mse_net)
print("Overall Average RMSE: ", overall_average_rmse_net)
print("Overall Average MAE: ", overall_average_mae_net)
print("Overall Average MAPE: ", overall_average_mape_net)

# Denormalize the errors for July
hourly_mse_july_net = np.array(hourly_mse_net[:31*24]) * std_net_consumption + mean_net_consumption
hourly_rmse_july_net = np.array(hourly_rmse_net[:31*24]) * std_net_consumption + mean_net_consumption
hourly_mae_july_net = np.array(hourly_mae_net[:31*24]) * std_net_consumption + mean_net_consumption
hourly_mape_july_net = np.array(hourly_mape_net[:31*24])* std_net_consumption / mean_net_consumption 
# Denormalize the errors for August
hourly_mse_august_net = np.array(hourly_mse_net[31*24:]) * std_net_consumption + mean_net_consumption
hourly_rmse_august_net = np.array(hourly_rmse_net[31*24:]) * std_net_consumption + mean_net_consumption
hourly_mae_august_net = np.array(hourly_mae_net[31*24:]) * std_net_consumption + mean_net_consumption
hourly_mape_august_net = np.array(hourly_mape_net[31*24:])* std_net_consumption / mean_net_consumption 

# Reshape the arrays for July into (24, 31) arrays
hourly_rmse_july_net_reshaped = hourly_rmse_july_net.reshape((24, 31), order='F')
hourly_mse_july_net_reshaped = hourly_mse_july_net.reshape((24, 31), order='F')
hourly_mae_july_net_reshaped = hourly_mae_july_net.reshape((24, 31), order='F')
hourly_mape_july_net_reshaped = hourly_mape_july_net.reshape((24, 31), order='F')
# Calculate the mean for each hour across all Mondays, Tuesdays, etc.
mean_rmse_per_hour_across_days_net = np.mean(hourly_rmse_july_net_reshaped, axis=1)
mean_mse_per_hour_across_days_net = np.mean(hourly_mse_july_net_reshaped, axis=1)
mean_mae_per_hour_across_days_net = np.mean(hourly_mae_july_net_reshaped, axis=1)
mean_mape_per_hour_across_days_net = np.mean(hourly_mape_july_net_reshaped, axis=1)
# Create DataFrames for each metric
rmse_df_net_july = pd.DataFrame({'Hour': range(24),
                        'Mean_RMSE_Monday': hourly_rmse_july_net_reshaped[:, 0],
                        'Mean_RMSE_Tuesday': hourly_rmse_july_net_reshaped[:, 1],
                        'Mean_RMSE_Wednesday': hourly_rmse_july_net_reshaped[:, 2],
                        'Mean_RMSE_Thursday': hourly_rmse_july_net_reshaped[:, 3],
                        'Mean_RMSE_Friday': hourly_rmse_july_net_reshaped[:, 4],
                        'Mean_RMSE_Saturday': hourly_rmse_july_net_reshaped[:, 5],
                        'Mean_RMSE_Sunday': hourly_rmse_july_net_reshaped[:, 6]})

mse_df_net_july = pd.DataFrame({'Hour': range(24),
                       'Mean_MSE_Monday': hourly_mse_july_net_reshaped[:, 0],
                       'Mean_MSE_Tuesday': hourly_mse_july_net_reshaped[:, 1],
                       'Mean_MSE_Wednesday': hourly_mse_july_net_reshaped[:, 2],
                       'Mean_MSE_Thursday': hourly_mse_july_net_reshaped[:, 3],
                       'Mean_MSE_Friday': hourly_mse_july_net_reshaped[:, 4],
                       'Mean_MSE_Saturday': hourly_mse_july_net_reshaped[:, 5],
                       'Mean_MSE_Sunday': hourly_mse_july_net_reshaped[:, 6]})

mae_df_net_july = pd.DataFrame({'Hour': range(24),
                       'Mean_MAE_Monday': hourly_mae_july_net_reshaped[:, 0],
                       'Mean_MAE_Tuesday': hourly_mae_july_net_reshaped[:, 1],
                       'Mean_MAE_Wednesday': hourly_mae_july_net_reshaped[:, 2],
                       'Mean_MAE_Thursday': hourly_mae_july_net_reshaped[:, 3],
                       'Mean_MAE_Friday': hourly_mae_july_net_reshaped[:, 4],
                       'Mean_MAE_Saturday': hourly_mae_july_net_reshaped[:, 5],
                       'Mean_MAE_Sunday': hourly_mae_july_net_reshaped[:, 6]})

mape_df_net_july = pd.DataFrame({'Hour': range(24),
                            'Mean_MAPE_Monday': hourly_mape_july_net_reshaped[:, 0],
                            'Mean_MAPE_Tuesday': hourly_mape_july_net_reshaped[:, 1],
                            'Mean_MAPE_Wednesday': hourly_mape_july_net_reshaped[:, 2],
                            'Mean_MAPE_Thursday': hourly_mape_july_net_reshaped[:, 3],
                            'Mean_MAPE_Friday': hourly_mape_july_net_reshaped[:, 4],
                            'Mean_MAPE_Saturday': hourly_mape_july_net_reshaped[:, 5],
                            'Mean_MAPE_Sunday': hourly_mape_july_net_reshaped[:, 6]})


# Reshape the arrays for August into (24, 31) arrays
hourly_rmse_august_net_reshaped = hourly_rmse_august_net.reshape((24, 31), order='F')
hourly_mse_august_net_reshaped = hourly_mse_august_net.reshape((24, 31), order='F')
hourly_mae_august_net_reshaped = hourly_mae_august_net.reshape((24, 31), order='F')
hourly_mape_august_net_reshaped = hourly_mape_august_net.reshape((24, 31), order='F')
# Calculate the mean for each hour across all Mondays, Tuesdays, etc.
mean_rmse_per_hour_across_days_august_net = np.mean(hourly_rmse_august_net_reshaped, axis=1)
mean_mse_per_hour_across_days_august_net = np.mean(hourly_mse_august_net_reshaped, axis=1)
mean_mae_per_hour_across_days_august_net = np.mean(hourly_mae_august_net_reshaped, axis=1)
mean_mape_per_hour_across_days_august_net = np.mean(hourly_mape_august_net_reshaped, axis=1)
# Create DataFrames for each metric
rmse_df_net_august = pd.DataFrame({'Hour': range(24),
                                   'Mean_RMSE_Monday': hourly_rmse_august_net_reshaped[:, 0],
                                   'Mean_RMSE_Tuesday': hourly_rmse_august_net_reshaped[:, 1],
                                   'Mean_RMSE_Wednesday': hourly_rmse_august_net_reshaped[:, 2],
                                   'Mean_RMSE_Thursday': hourly_rmse_august_net_reshaped[:, 3],
                                   'Mean_RMSE_Friday': hourly_rmse_august_net_reshaped[:, 4],
                                   'Mean_RMSE_Saturday': hourly_rmse_august_net_reshaped[:, 5],
                                   'Mean_RMSE_Sunday': hourly_rmse_august_net_reshaped[:, 6]})

mse_df_net_august = pd.DataFrame({'Hour': range(24),
                                  'Mean_MSE_Monday': hourly_mse_august_net_reshaped[:, 0],
                                  'Mean_MSE_Tuesday': hourly_mse_august_net_reshaped[:, 1],
                                  'Mean_MSE_Wednesday': hourly_mse_august_net_reshaped[:, 2],
                                  'Mean_MSE_Thursday': hourly_mse_august_net_reshaped[:, 3],
                                  'Mean_MSE_Friday': hourly_mse_august_net_reshaped[:, 4],
                                  'Mean_MSE_Saturday': hourly_mse_august_net_reshaped[:, 5],
                                  'Mean_MSE_Sunday': hourly_mse_august_net_reshaped[:, 6]})

mae_df_net_august = pd.DataFrame({'Hour': range(24),
                                  'Mean_MAE_Monday': hourly_mae_august_net_reshaped[:, 0],
                                  'Mean_MAE_Tuesday': hourly_mae_august_net_reshaped[:, 1],
                                  'Mean_MAE_Wednesday': hourly_mae_august_net_reshaped[:, 2],
                                  'Mean_MAE_Thursday': hourly_mae_august_net_reshaped[:, 3],
                                  'Mean_MAE_Friday': hourly_mae_august_net_reshaped[:, 4],
                                  'Mean_MAE_Saturday': hourly_mae_august_net_reshaped[:, 5],
                                  'Mean_MAE_Sunday': hourly_mae_august_net_reshaped[:, 6]})

mape_df_net_august = pd.DataFrame({'Hour': range(24),
                                  'Mean_MAPE_Monday': hourly_mape_august_net_reshaped[:, 0],
                                  'Mean_MAPE_Tuesday': hourly_mape_august_net_reshaped[:, 1],
                                  'Mean_MAPE_Wednesday': hourly_mape_august_net_reshaped[:, 2],
                                  'Mean_MAPE_Thursday': hourly_mape_august_net_reshaped[:, 3],
                                  'Mean_MAPE_Friday': hourly_mape_august_net_reshaped[:, 4],
                                  'Mean_MAPE_Saturday': hourly_mape_august_net_reshaped[:, 5],
                                  'Mean_MAPE_Sunday': hourly_mape_august_net_reshaped[:, 6]})



# Calculate the average MSE, RMSE, and MAE for July and August
average_mse_july_net = np.mean(hourly_mse_july_net)
average_rmse_july_net = np.mean(hourly_rmse_july_net)
average_mae_july_net = np.mean(hourly_mae_july_net)
average_mape_july_net = np.mean(hourly_mape_july_net)
average_mse_august_net = np.mean(hourly_mse_august_net)
average_rmse_august_net = np.mean(hourly_rmse_august_net)
average_mae_august_net = np.mean(hourly_mae_august_net)
average_mape_august_net = np.mean(hourly_mape_august_net)


######## Store error values ########

######## Store error values ########

normalized_real_net_consumption = np.array(real_net_consumption_values)* std_net_consumption + mean_net_consumption
denormalized_net_forecast_values = np.array(net_forecast_values) * std_net_consumption + mean_net_consumption
# Denormalize the forecasted values
forecasted_loads_denormalized_net = np.array(forecasted_loads_net) * std_net_consumption + mean_net_consumption


mean_solar = solar_data['PV25'].mean()  # replace with actual mean
std_solar = solar_data['PV25'].std()    # replace with actual std
mean_consumption = data_real_resampled['MT_080_normalized'].mean()  # Replace 'Load_Column_Name' with your actual column name for load
std_consumption = data_real_resampled['MT_080_normalized'].std() 

denormalized_forecast_solar= np.array(forecast_solar_df)* std_solar + mean_solar
denormalized_forecast_load = np.array(forecasted_loads_df)* std_consumption + mean_consumption

denormalized_real_consumption = np.array(real_load)* std_consumption + mean_consumption
denormalized_real_solar = np.array(real_solar)* std_solar + mean_solar

normalized_real_solar = np.array(real_solar)* std_solar + mean_solar
normlized_real_load= np.array(real_load)* std_consumption + mean_consumption

overall_rmse_solar = np.array(hourly_rmse_solar) * std_solar + mean_solar
overall_rmse_load = np.array(hourly_rmse) * std_consumption + mean_consumption


#solar
# Create DataFrames for each sheet
columns = ["DateTime", "Forecasted_Values", "Hourly_RMSE"]

# Method 1 - solar
method1_df = pd.DataFrame(columns=["DateTime","Forecasted_Values", "Hourly_RMSE"])
method1_df["DateTime"] = forecast_solar_df.index
method1_df["Forecasted_Values"] = denormalized_forecast_solar
method1_df["Hourly_RMSE"] = overall_rmse_solar

# Real Values - solar
real_values_df = pd.DataFrame(columns=["DateTime", "Real_SOLAR"])
real_values_df["DateTime"] = forecasted_loads_df.index
real_values_df["Real_SOLAR"] = normalized_real_solar
# Create a writer
with pd.ExcelWriter('SARIMA_scores_verao_solar.xlsx') as writer:
    # Write each DataFrame to a different sheet
    method1_df.to_excel(writer, sheet_name='Method1', index=False)
    real_values_df.to_excel(writer, sheet_name='Real_Values', index=False)
    

######cosncuption

# Create DataFrames for each sheet
columns = ["DateTime", "Forecasted_Values", "Hourly_RMSE"]

# Method 1 - Consumption
method1_df = pd.DataFrame(columns=["DateTime","Forecasted_Values", "Hourly_RMSE"])
method1_df["DateTime"] = forecast_solar_df.index
method1_df["Forecasted_Values"] = denormalized_forecast_load
method1_df["Hourly_RMSE"] = overall_rmse_load

# Real Values -Consumption
real_values_df = pd.DataFrame(columns=["DateTime", "Real_Consumption"])
real_values_df["DateTime"] = forecasted_loads_df.index
real_values_df["Real_Consumption"] = normlized_real_load
# Create a writer
with pd.ExcelWriter('SARIMA_scores_verao_load.xlsx') as writer:
    # Write each DataFrame to a different sheet
    method1_df.to_excel(writer, sheet_name='Method1', index=False)
    real_values_df.to_excel(writer, sheet_name='Real_Values', index=False)














# # Create DataFrames for each sheet
# columns = ["DateTime", "Forecasted_Values", "Hourly_RMSE", "Hourly_MSE", "Hourly_MAE"]

# # Method 1 - Net Consumption
# method1_df = pd.DataFrame(columns=["DateTime","Forecasted_Values", "Hourly_RMSE", "Hourly_MSE", "Hourly_MAE"])
# method1_df["DateTime"] = forecasted_loads_df.index
# method1_df["Forecasted_Values"] = denormalized_net_forecast_values
# method1_df["Hourly_RMSE"] = overall_rmse_nl1
# method1_df["Hourly_MSE"] =overall_mse_nl1
# method1_df["Hourly_MAE"] = overall_mae_nl1
# method1_df["Hourly_MAPE"] = overall_mape_nl1
# # Method 2 - Net Load
# method2_df = pd.DataFrame(columns=["DateTime","Forecasted_Values", "Hourly_RMSE", "Hourly_MSE", "Hourly_MAE"])

# # Flatten the nested arrays in forecasted_loads_net

# method2_df["DateTime"] = forecasted_loads_df.index
# method2_df["Forecasted_Values"] = forecasted_loads_denormalized_net
# method2_df["Hourly_RMSE"] = overall_rmse_net
# method2_df["Hourly_MSE"] = overall_mse_net
# method2_df["Hourly_MAE"] = overall_mae_net
# method2_df["Hourly_MAPE"] = overall_mape_net

# # Real Values - Net Consumption
# real_values_df = pd.DataFrame(columns=["DateTime", "Real_Net_Consumption"])
# real_values_df["DateTime"] = forecasted_loads_df.index
# real_values_df["Real_Net_Consumption"] = normalized_real_net_consumption 
# # Create a writer
# with pd.ExcelWriter('SARIMA_scores_verao.xlsx') as writer:
#     # Write each DataFrame to a different sheet
#     method1_df.to_excel(writer, sheet_name='Method1', index=False)
#     method2_df.to_excel(writer, sheet_name='Method2', index=False)
#     real_values_df.to_excel(writer, sheet_name='Real_Values', index=False)
    
    
    
# # Create DataFrames for Method 1 and Method 2 results
# method1_df = pd.DataFrame(index=['July', 'August'], columns=['RMSE', 'MSE', 'MAE'])
# method2_df = pd.DataFrame(index=['July', 'August'], columns=['RMSE', 'MSE', 'MAE'])

# # Method 1 results
# method1_df.loc['July', 'RMSE'] = average_rmse_july_nl1
# method1_df.loc['August', 'RMSE'] = average_rmse_august_nl1
# method1_df.loc['July', 'MSE'] = average_mse_july_nl1
# method1_df.loc['August', 'MSE'] = average_mse_august_nl1
# method1_df.loc['July', 'MAE'] = average_mae_july_nl1
# method1_df.loc['August', 'MAE'] = average_mae_august_nl1
# method1_df.loc['July', 'MAPE'] = average_mape_july_nl1
# method1_df.loc['August', 'MAPE'] = average_mape_august_nl1
# # Method 2 results
# method2_df.loc['July', 'RMSE'] = average_rmse_july_net 
# method2_df.loc['August', 'RMSE'] = average_rmse_august_net
# method2_df.loc['July', 'MSE'] = average_mse_july_net  
# method2_df.loc['August', 'MSE'] = average_mse_august_net
# method2_df.loc['July', 'MAE'] = average_mae_july_net
# method2_df.loc['August', 'MAE'] = average_mae_august_net
# method2_df.loc['July', 'MAPE'] = average_mape_july_net
# method2_df.loc['August', 'MAPE'] = average_mape_august_net


# # Create a writer
# with pd.ExcelWriter('Montly_SARIMA_scores_verao.xlsx') as writer:
#     # Write each DataFrame to a different sheet
#     method1_df.to_excel(writer, sheet_name='Method1', index=True)
#     method2_df.to_excel(writer, sheet_name='Method2', index=True)
    
# # Create a writer
# with pd.ExcelWriter('SARIMA_weekday_scores_verao.xlsx') as writer:
#     rmse_df_july.to_excel(writer, sheet_name='July_weekdays_m1_RMSE', index=False)
#     mse_df_july.to_excel(writer, sheet_name='July_weekdays_m1_MSE', index=False)
#     mae_df_july.to_excel(writer, sheet_name='July_weekdays_m1_MAE', index=False)
#     mape_df_july.to_excel(writer, sheet_name='July_weekdays_m1_MAPE', index=False)
#     rmse_df_net_july.to_excel(writer, sheet_name='July_weekdays_m2_RMSE', index=False)
#     mse_df_net_july.to_excel(writer, sheet_name='July_weekdays_m2_MSE', index=False)
#     mae_df_net_july.to_excel(writer, sheet_name='July_weekdays_m2_MAE', index=False)
#     mape_df_net_july.to_excel(writer, sheet_name='July_weekdays_m2_MAPE', index=False)
#     rmse_df_august.to_excel(writer, sheet_name='August_weekdays_m1_RMSE', index=False)
#     mse_df_august.to_excel(writer, sheet_name='August_weekdays_m1_MSE', index=False)
#     mae_df_august.to_excel(writer, sheet_name='August_weekdays_m1_MAE', index=False)
#     mape_df_august.to_excel(writer, sheet_name='August_weekdays_m1_MAPE', index=False)
#     rmse_df_net_august.to_excel(writer, sheet_name='August_weekdays_m2_RMSE', index=False)
#     mse_df_net_august.to_excel(writer, sheet_name='August_weekdays_m2_MSE', index=False)
#     mae_df_net_august.to_excel(writer, sheet_name='August_weekdays_m2_MAE', index=False)
#     mape_df_net_august.to_excel(writer, sheet_name='August_weekdays_m2_MAPE', index=False)



