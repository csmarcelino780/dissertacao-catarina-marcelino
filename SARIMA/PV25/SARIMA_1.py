# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:25:09 2023

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
real_load_prediction_days = solar_real_resampled.loc[solar_real_resampled.index.month.isin([7, 8])]['PV25']
real_solar_prediction_days = solar_real_resampled.loc[solar_real_resampled.index.month.isin([7, 8])]['PV25']  # Assuming solar data is used for solar_real

# Calculate the real net consumption for the selected days
real_net_consumption = real_load_prediction_days - real_solar_prediction_days

############################## method 1 ( forecasted load - forecasted solar)#############################
months_to_calculate = [7, 8]  # July and August
days_in_month = [31, 31]  

# Initialize lists to store net consumption, MSE, and RMSE for each day
net_consumption_per_day = []
mse_per_day = []
rmse_per_day = []
# Initialize lists to store RMSE and MSE per weekday
rmse_per_weekday_net = [0] * 7
mse_per_weekday_net = [0] * 7
count_per_weekday_net = [0] * 7

# Loop over months, days, and hours
for month in [7, 8]:
    for day in range(1, 32):
        for hour in range(24):
            prediction_datetime = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")
            
            # Check if the prediction date is within the data range
            if prediction_datetime >= real_load_prediction_days.index[0]:
                # Make load and solar forecasts
                load_forecast = sarima_fit.predict(start=len(real_load_prediction_days), end=len(real_load_prediction_days))
                solar_forecast = sarima_solar_fit.predict(start=len(real_solar_prediction_days), end=len(real_solar_prediction_days))
                
                # Calculate net load by subtracting solar forecast from load forecast
                net_load = load_forecast - solar_forecast
                
                # Calculate real net consumption for the current day and hour
                real_net_consumption_val = real_net_consumption.loc[prediction_datetime]
                
                # Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for the current hour
                mse = mean_squared_error([real_net_consumption_val], [net_load])
                rmse = np.sqrt(mse)
                
                # Append the calculated MSE and RMSE to the respective lists
                mse_per_day.append(mse)
                rmse_per_day.append(rmse)
                
                # Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for the current hour
                mse_net = mean_squared_error([real_net_consumption_val], [net_load])
                rmse_net = np.sqrt(mse_net)
                
                # Calculate the weekday (0 = Monday, 6 = Sunday)
                weekday = prediction_datetime.weekday()

                # Accumulate RMSE and MSE per weekday
                rmse_per_weekday_net[weekday] += rmse_net
                mse_per_weekday_net[weekday] += mse_net
                count_per_weekday_net[weekday] += 1

# Calculate the average RMSE and MSE per weekday
average_rmse_per_weekday_net = [rmse / count if count > 0 else 0 for rmse, count in zip(rmse_per_weekday_net, count_per_weekday_net)]
average_mse_per_weekday_net = [mse / count if count > 0 else 0 for mse, count in zip(mse_per_weekday_net, count_per_weekday_net)]

# Output the average RMSE and MSE per weekday for net load
weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for i, day in enumerate(weekday_names):
    print(f"Average RMSE for {day} (Net Load): {average_rmse_per_weekday_net[i]:.2f}")
    print(f"Average MSE for {day} (Net Load): {average_mse_per_weekday_net[i]:.2f}")


# Calculate overall MSE and RMSE for all the days in July and August
overall_mse = np.mean(mse_per_day)
overall_rmse = np.mean(rmse_per_day)

# Calculate hourly MSE and RMSE for July and August
hourly_mse_july_nl = mse_per_day[:31 * 24]  # Assuming July has 31 days
hourly_rmse_july_nl = rmse_per_day[:31 * 24]
hourly_mse_august_nl = mse_per_day[31 * 24:]  # The rest is for August
hourly_rmse_august_nl = rmse_per_day[31 * 24:]

# Calculate the average MSE and RMSE for July and August
average_mse_july_nl = np.mean(hourly_mse_july_nl)
average_rmse_july_nl = np.mean(hourly_rmse_july_nl)
average_mse_august_nl = np.mean(hourly_mse_august_nl)
average_rmse_august_nl = np.mean(hourly_rmse_august_nl)

# Print the overall MSE and RMSE
print(f"Overall Mean Squared Error (MSE) for net consumption in July and August: {overall_mse}")
print(f"Overall Root Mean Squared Error (RMSE) for net consumption in July and August: {overall_rmse}")

# Print the average MSE and RMSE for July and August
print(f"Average Mean Squared Error (MSE) for net consumption in July: {average_mse_july_nl}")
print(f"Average Root Mean Squared Error (RMSE) for net consumption in July: {average_rmse_july_nl}")
print(f"Average Mean Squared Error (MSE) for net consumption in August: {average_mse_august_nl}")
print(f"Average Root Mean Squared Error (RMSE) for net consumption in August: {average_rmse_august_nl}")


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

# Initialize lists to store RMSE and MSE per weekday
rmse_per_weekday_net = [0] * 7
mse_per_weekday_net = [0] * 7
count_per_weekday_net = [0] * 7


# Months to forecast
months = [7, 8]  # July and August
days_in_month = [31, 31]  # Number of days in July and August, adjust as needed

monthly_rmse = {month: [] for month in months}
monthly_mse = {month: [] for month in months}

# Convert end_date to a Timestamp
end_date = pd.to_datetime(end_date)

hourly_rmse_net = []
hourly_mse_net = []

forecasted_loads_net = []

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

            # Append the forecasted load and timestamp to the results
            forecasted_loads_net.append((prediction_datetime_net, forecasted_load_net[0]))

            # Extract the observed values for the same time from net_load
            observed_values_net = net_load.loc[prediction_datetime_net].values

            # Calculate RMSE for the current hour
            rmse_net = np.sqrt(mean_squared_error(observed_values_net, forecasted_load_net))
            hourly_rmse_net.append(rmse_net)

            # Calculate MSE for the current hour
            mse_net = mean_squared_error(observed_values_net, forecasted_load_net)
            hourly_mse_net.append(mse_net)


            # Calculate the weekday (0 = Monday, 6 = Sunday)
            weekday = prediction_datetime_net.weekday()

            # Accumulate RMSE and MSE per weekday
            rmse_per_weekday_net[weekday] += rmse_net
            mse_per_weekday_net[weekday] += mse_net
            count_per_weekday_net[weekday] += 1

# Calculate the average RMSE and MSE per weekday
average_rmse_per_weekday_net = [rmse / count if count > 0 else 0 for rmse, count in zip(rmse_per_weekday_net, count_per_weekday_net)]
average_mse_per_weekday_net = [mse / count if count > 0 else 0 for mse, count in zip(mse_per_weekday_net, count_per_weekday_net)]

# Output the average RMSE and MSE per weekday for net load
weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for i, day in enumerate(weekday_names):
    print(f"Average RMSE for {day} (Net Load): {average_rmse_per_weekday_net[i]:.2f}")
    print(f"Average MSE for {day} (Net Load): {average_mse_per_weekday_net[i]:.2f}")

# Calculate monthly RMSE for net load by taking the average for each month
july_rmse_net = np.mean(hourly_rmse_net[:31 * 24])  # RMSE for July for net load
august_rmse_net = np.mean(hourly_rmse_net[31 * 24:])  # RMSE for August for net load
july_mse_net = np.mean(hourly_mse_net[:31 * 24])  # MSE for July for net load
august_mse_net = np.mean(hourly_mse_net[31 * 24:])  # MSE for August for net load

print("Average RMSE for July (Net Load):", july_rmse_net)
print("Average RMSE for August (Net Load):", august_rmse_net)
print("Average MSE for July (Net Load):", july_mse_net)
print("Average MSE for August (Net Load):", august_mse_net)
