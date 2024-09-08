# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:11:42 2023

@author: catar
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import holidays


def select_best_cluster_for_feature(feature_data, data_resampled, X_common, y, k):
    best_cluster_metric = np.inf
    best_cluster_name = None

    cluster_columns = [col for col in feature_data.columns if col != 'Timestamp']
    print("Feature Data columns:", feature_data.columns)

    for cluster_name in cluster_columns:
        cluster_data = feature_data[['Timestamp', cluster_name]]
        cluster_data.set_index('Timestamp', inplace=True)
        cluster_data.index = pd.to_datetime(cluster_data.index, errors='coerce')
        cluster_data = cluster_data.dropna()

        common_index = cluster_data.index.intersection(data_resampled.index)
        cluster_data_resampled = cluster_data.loc[common_index]
        X_cluster = data_resampled.loc[common_index]

        fold_rmse_scores = []
        
        k_neighbors = 5
        tscv = TimeSeriesSplit(n_splits=k)

        for train_index, test_index in tscv.split(X_cluster):
            X_train, X_test = X_cluster.iloc[train_index], X_cluster.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Use k-NN model for forecasting
            knn_model = KNeighborsRegressor(n_neighbors=k_neighbors)  # Define k-NN model
            knn_model.fit(X_train, y_train)

            forecast_features = X_test.iloc[-24:]

            forecasted_load = knn_model.predict(forecast_features)

            rmse = np.sqrt(mean_squared_error(y_test[-24:], forecasted_load))
            fold_rmse_scores.append(rmse)

        avg_rmse = np.mean(fold_rmse_scores)

        if avg_rmse < best_cluster_metric:
            best_cluster_metric = avg_rmse
            best_cluster_name = cluster_name

    return best_cluster_name


    best_cluster_metric = np.inf
    best_cluster_name = None

    cluster_columns = [col for col in feature_data.columns if col != 'Timestamp']
    print("Feature Data columns:", feature_data.columns)

    for cluster_name in cluster_columns:
        cluster_data = feature_data[['Timestamp', cluster_name]]
        cluster_data.set_index('Timestamp', inplace=True)
        cluster_data.index = pd.to_datetime(cluster_data.index, errors='coerce')
        cluster_data = cluster_data.dropna()

        common_index = cluster_data.index.intersection(data_resampled.index)
        cluster_data_resampled = cluster_data.loc[common_index]
        X_cluster = data_resampled.loc[common_index]

        fold_rmse_scores = []

        tscv = TimeSeriesSplit(n_splits=k)

        for train_index, test_index in tscv.split(X_cluster):
            X_train, X_test = X_cluster.iloc[train_index], X_cluster.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Create a k-NN model with the desired number of neighbors (e.g., 5)
            k_neighbors = 5
            knn_model = KNeighborsRegressor(n_neighbors=k_neighbors)
            knn_model.fit(X_train, y_train)

            forecast_features = X_test.iloc[-24:]

            forecasted_load = knn_model.predict(forecast_features)

            rmse = np.sqrt(mean_squared_error(y_test[-24:], forecasted_load))
            fold_rmse_scores.append(rmse)

        avg_rmse = np.mean(fold_rmse_scores)

        if avg_rmse < best_cluster_metric:
            best_cluster_metric = avg_rmse
            best_cluster_name = cluster_name

    return best_cluster_name

def cluster_features_for_data(clima_file, X, k):
    best_clusters = {}
    cluster_values_df = pd.DataFrame(index=X.index)

    for feature_name in pd.ExcelFile(clima_file).sheet_names:
        feature_data = pd.read_excel(clima_file, sheet_name=feature_name)
        
        # Filter feature_data to only include common dates
        common_dates = feature_data['Timestamp'].astype('datetime64[ns]').isin(X.index)
        feature_data = feature_data[common_dates]
        
        best_cluster_name = select_best_cluster_for_feature(feature_data, data_resampled, X, y, k)
        best_clusters[feature_name] = best_cluster_name

    for feature_name, cluster_name in best_clusters.items():
        feature_data = pd.read_excel(clima_file, sheet_name=feature_name)
        feature_data.set_index('Timestamp', inplace=True)
        feature_data.index = pd.to_datetime(feature_data.index, errors='coerce')
        
        # Filter feature_data to only include common dates
        common_dates = feature_data.index.isin(X.index)
        feature_data = feature_data[common_dates]
        
        if cluster_name in feature_data.columns:
            cluster_values_df[feature_name + "_Cluster"] = feature_data[cluster_name]

    return cluster_values_df

    mse_scores = []
    rmse_scores = []
    
    tscv = TimeSeriesSplit(n_splits=k)
    knn = KNeighborsRegressor(n_neighbors=5)  # Adjust the number of neighbors as needed

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply the date range filter to both X_train and y_train
        common_dates = (X_train.index >= start_date) & (X_train.index <= end_date)
        X_train = X_train[common_dates]
        y_train = y_train[common_dates]

        knn.fit(X_train[all_features], y_train)

        forecast_features = X_test[all_features].iloc[-24:]
        forecasted_values = knn.predict(forecast_features)

        mse = mean_squared_error(y_test[-24:], forecasted_values)
        mse_scores.append(mse)

        rmse = np.sqrt(mse)
        rmse_scores.append(rmse)
    
    return mse_scores, rmse_scores

def forward_feature_selection(X, y, feature_names, k):
    best_features = []
    best_metric = np.inf

    for feature in feature_names:
        temp_features = best_features + [feature]
        
        # Debugging: Print available columns and selected features
        print("Columns in X:", X.columns)
        print("Selected Features:", temp_features)
        
        X_selected = X[temp_features]

        fold_rmse_scores = []

        tscv = TimeSeriesSplit(n_splits=k)

        for train_index, test_index in tscv.split(X_selected):
            X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            knn = KNeighborsRegressor(n_neighbors=5)  # Adjust the number of neighbors as needed
            knn.fit(X_train, y_train)

            forecast_features = X_test.iloc[-24:]

            forecasted_load = knn.predict(forecast_features)

            rmse = np.sqrt(mean_squared_error(y_test[-24:], forecasted_load))
            fold_rmse_scores.append(rmse)

        avg_rmse = np.mean(fold_rmse_scores)

        if avg_rmse < best_metric:
            best_metric = avg_rmse
            best_features.append(feature)
    # Removed the else block that stopped the loop

    return best_features



# Load  data
start_date = '2012-02-01'  # where trainning starts 
end_date = '2013-06-29'  # where traning ends 
k = 5

#clima file 
clima_file = r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Clustering de dados climaticos\mean_sheets_with_date.xlsx"

# consumption file 
data_df = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Normalizaçao Consumo\Normalized_Demand_Tese_teste.csv', parse_dates=['DateTime'], usecols=["MT_080_normalized", "DateTime"])

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

# store the best clusters and features
best_clusters = {}
best_features = []

# Empty DataFrame with an index based on the non-NaN rows in X
cluster_values_df = pd.DataFrame(index=X.index)
# Call cluster_features_for_data
cluster_values_df_load = cluster_features_for_data(clima_file, X, k)
# Merge cluster_values_df with 'X' using a common index
X = X.join(cluster_values_df)
X.dropna(inplace=True) # Drop NaN rows in X
valid_indices = X.index # Update the valid_indices


valid_positions = [X.index.get_loc(idx) for idx in valid_indices]
# Use those positions to filter y
y = y[valid_positions]

# Add features
X['DayOfWeek'] = X.index.dayofweek
X['Day'] = X.index.day
X['Month'] = X.index.month
X['Hour'] = X.index.hour
pt_holidays = holidays.Portugal()
X['IsHoliday'] = X.index.map(lambda x: int(x in pt_holidays))

# Perform forward feature selection
best_features = forward_feature_selection(X, y, list(X.columns), k)
# Include time-based features in the model training
all_features = best_features + ['DayOfWeek', 'Day', 'Month', 'Hour', 'IsHoliday']

# K-Nearest Neighbors (k-NN) model
k_neighbors = 5  # Choose an appropriate number of neighbors
knn_model = KNeighborsRegressor(n_neighbors=k_neighbors)

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=k)

# Store evaluation metrics
mse_scores = []
rmse_scores = []

# Train and evaluate the model using cross-validation

for train_index, test_index in tscv.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply the date range filter to both X_train and y_train
    common_dates = (X_train.index >= start_date) & (X_train.index <= end_date)
    X_train = X_train[common_dates]
    y_train = y_train[common_dates]
    
    # Fit the k-NN model 
    knn_model.fit(X_train[all_features], y_train)

    # Forecast
    forecast_features = X_test[all_features].iloc[-24:]
    forecasted_load = knn_model.predict(forecast_features)

    # Evaluate the model
    mse = mean_squared_error(y_test[-24:], forecasted_load)
    mse_scores.append(mse)

    rmse = np.sqrt(mse)
    rmse_scores.append(rmse)

# Calculate average MSE and RMSE
avg_mse = np.mean(mse_scores)
avg_rmse = np.mean(rmse_scores)

# Output the evaluation metrics
print(f"Average MSE: {avg_mse}")
print(f"Average RMSE: {avg_rmse}")

forecasted_loads = []


# Months to forcast 
months = [7, 8]  # July and August
days_in_month = [31, 31]  # Number of days in July and August, adjust as needed

monthly_rmse = {month: [] for month in months}
monthly_mse = {month: [] for month in months}

# Convert end_date to a Timestamp
end_date = pd.to_datetime(end_date)

hourly_rmse = []
hourly_mse = []  

# Make day-ahead forecasts for all the hours of each day
for month in [7, 8]:  # July and August of 2013
    for day in range(1, 32):
        for hour in range(24):
            prediction_datetime = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")
            
            # Prediction day is after the end_date for training
            if prediction_datetime > end_date or (hour >= 12 and hour < 24 and day > 1):
                # Select relevant features for the prediction time
                forecast_features = X[all_features].loc[prediction_datetime]
                
                # Make the hour-ahead forecast using k-NN
                forecasted_load = knn_model.predict(forecast_features.values.reshape(1, -1))
                
                # Append the forecasted load to the results
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
forecast_df = pd.DataFrame(forecasted_loads, columns=["DateTime", "Forecasted_Load"])
forecast_df.set_index("DateTime", inplace=True)

# Calculate monthly RMSE by taking the average for each month
july_rmse = np.mean(hourly_rmse[:31 * 24])  # RMSE for July
august_rmse = np.mean(hourly_rmse[31 * 24:])  # RMSE for August
july_mse = np.mean(hourly_mse[:31 * 24])  # MSE for July
august_mse = np.mean(hourly_mse[31 * 24:])  # MSE for August

print("Average RMSE for July:", july_rmse)
print("Average RMSE for August:", august_rmse)



############################################ sOLAR ################################################
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

# store the best clusters and features
best_clusters_solar = {}
best_features_solar = []

# Empty DataFrame with an index based on the non-NaN rows in X
cluster_values_df_solar = pd.DataFrame(index=X_solar.index)
# Call cluster_features_for_data
cluster_values_df_solar = cluster_features_for_data(clima_file, X_solar, k)
# Merge cluster_values_df with 'X' using a common index
X_solar = X_solar.join(cluster_values_df_solar)
X_solar.dropna(inplace=True) # Drop NaN rows in X
valid_indices_solar = X_solar.index # Update the valid_indices

valid_positions_solar = [X_solar.index.get_loc(idx) for idx in valid_indices_solar]
# Use those positions to filter y_solar
y_solar = y_solar[valid_positions_solar]

# Add features
X_solar['DayOfWeek'] = X_solar.index.dayofweek
X_solar['Day'] = X_solar.index.day
X_solar['Month'] = X_solar.index.month
X_solar['Hour'] = X_solar.index.hour
pt_holidays = holidays.Portugal()
X_solar['IsHoliday'] = X_solar.index.map(lambda x: int(x in pt_holidays))

# Perform forward feature selection for solar data
best_features_solar = forward_feature_selection(X_solar, y_solar, list(X_solar.columns), k)
# Include time-based features in the model training for solar data
all_features_solar = best_features_solar + ['DayOfWeek', 'Day', 'Month', 'Hour', 'IsHoliday']

# K-Nearest Neighbors (k-NN) model for solar data
knn_solar = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=k)

# Store evaluation metrics for solar data
mse_scores_solar = []
rmse_scores_solar = []

# Train and evaluate the model using cross-validation for solar data
for train_index_solar, test_index_solar in tscv.split(X_solar):
    
    X_train_solar, X_test_solar = X_solar.iloc[train_index_solar], X_solar.iloc[test_index_solar]
    y_train_solar, y_test_solar = y_solar[train_index_solar], y_solar[test_index_solar]

    # Apply the date range filter to both X_train_solar and y_train_solar
    common_dates_solar = (X_train_solar.index >= start_date) & (X_train_solar.index <= end_date)
    X_train_solar = X_train_solar[common_dates_solar]
    y_train_solar = y_train_solar[common_dates_solar]
    
    # Fit the k-NN model
    knn_solar.fit(X_train_solar[all_features_solar], y_train_solar)

    # Forecast
    forecast_features_solar = X_test_solar[all_features_solar].iloc[-24:]
    forecasted_solar = knn_solar.predict(forecast_features_solar)

    # Evaluate the model for solar data
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

for month in [7, 8]:  # July and August for solar data
    for day in range(1, 32):
        for hour in range(24):  
            prediction_datetime_solar = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")
        
            # prediction day is after the end_date for training for solar data
            if prediction_datetime_solar > end_date_solar or (hour >= 12 and hour < 24 and day > 1):
                # Select relevant features for the prediction time for solar data
                forecast_features_solar = X_solar[all_features_solar].loc[prediction_datetime_solar]
                
                # Make the hour-ahead forecast for solar data using k-NN
                forecasted_solar_values = knn_solar.predict(forecast_features_solar.values.reshape(1, -1))
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

#  store net consumption, MSE, and RMSE for each day
net_consumption_per_day = []
mse_per_day = []
rmse_per_day = []

for month in [7, 8]:  # July and August for solar data
    for day in range(1, 32):
        for hour in range(24):  # 24 hours in a day
            prediction_datetime_solar = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")
        
            #  prediction day should be excluded (e.g., before your data starts)
            if prediction_datetime_solar >= data_real_resampled.index[0]:
                # Calculate the real net consumption for the current day and hour
                real_load = real_load_prediction_days.loc[prediction_datetime_solar]
                real_solar = real_solar_prediction_days.loc[prediction_datetime_solar]
                real_net_consumption = real_load - real_solar

                # Append the calculated values to the respective lists
                net_consumption_per_day.append(real_net_consumption)

                # Calculate the net load forecast for the current day and hour
                net_forecast = forecast_df.loc[prediction_datetime_solar]["Forecasted_Load"] - forecast_solar_df.loc[prediction_datetime_solar]["Forecasted_Solar"]

                # Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for the current hour
                mse = mean_squared_error([real_net_consumption], [net_forecast])
                rmse = np.sqrt(mse)

                # Append the calculated MSE and RMSE to the respective lists
                mse_per_day.append(mse)
                rmse_per_day.append(rmse)

# Calculate the overall MSE and RMSE for all the days in July and August
overall_mse = np.mean(mse_per_day)
overall_rmse = np.mean(rmse_per_day)

# Calculate the hourly MSE and RMSE for July and August
hourly_mse_july_nl1 = mse_per_day[:31*24]  # Assuming July has 31 days
hourly_rmse_july_nl1 = rmse_per_day[:31*24]
hourly_mse_august_nl1 = mse_per_day[31*24:]  # The rest is for August
hourly_rmse_august_nl1 = rmse_per_day[31*24:]

# Calculate the average MSE and RMSE for July and August
average_mse_july_nl1 = np.mean(hourly_mse_july_nl1)
average_rmse_july_nl1 = np.mean(hourly_rmse_july_nl1)
average_mse_august_nl1 = np.mean(hourly_mse_august_nl1)
average_rmse_august_nl1 = np.mean(hourly_rmse_august_nl1)

# Print the overall MSE and RMSE
print(f"Overall Mean Squared Error (MSE) for net consumption in July and August: {overall_mse}")
print(f"Overall Root Mean Squared Error (RMSE) for net consumption in July and August: {overall_rmse}")

# Print the average MSE and RMSE for July and August
print(f"Average Mean Squared Error (MSE) for net consumption in July: {average_mse_july_nl1}")
print(f"Average Root Mean Squared Error (RMSE) for net consumption in July: {average_rmse_july_nl1}")
print(f"Average Mean Squared Error (MSE) for net consumption in August: {average_mse_august_nl1}")
print(f"Average Root Mean Squared Error (RMSE) for net consumption in August: {average_rmse_august_nl1}")




############################## method 2 #############################
# Load and preprocess net load and solar data
net_load = data_resampled['MT_080_normalized'] - solar_resampled['PV25']

if isinstance(net_load, pd.Series):
    net_load = pd.DataFrame(net_load, columns=['Net_Load'])

# Resampling (if needed)
net_load = net_load.resample('H').mean()

# Drop NaN values (if needed)
net_load.dropna(inplace=True)


X_net = net_load.iloc[:-24].dropna() # exclud the last 24 rows + dropping rows with NaN values
y_net = net_load.iloc[24:].values

#store the best clusters and features
best_clusters = {}
best_features = []

# Empty DataFrame with an index based on the non-NaN rows in X
cluster_values_df = pd.DataFrame(index=X_net.index)
# Call cluster_features_for_data
cluster_values_df_net_load = cluster_features_for_data(clima_file, X_net, k)

# Merge cluster_values_df with 'X' using a common index
X_net = X_net.join(cluster_values_df)
X_net.dropna(inplace=True) # Drop NaN rows in X
valid_indices = X_net.index # Update the valid_indice


valid_positions = [X_net.index.get_loc(idx) for idx in valid_indices]
# Use those positions to filter y_net
y_net = y_net[valid_positions]

# Add  features to X_net
X_net['DayOfWeek'] = X_net.index.dayofweek
X_net['Day'] = X_net.index.day
X_net['Month'] = X_net.index.month
X_net['Hour'] = X_net.index.hour
pt_holidays = holidays.Portugal()
X_net['IsHoliday'] = X_net.index.map(lambda x: int(x in pt_holidays))

# Perform forward feature selection
best_features_net = forward_feature_selection(X_net, y_net, list(X_net.columns), k)
# Include time-based features in the model training
all_features_net = best_features_net + ['DayOfWeek', 'Day', 'Month', 'Hour', 'IsHoliday']

# K-Nearest Neighbors model for net load
knn_net = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed

# TimeSeriesSplit for cross-validation
tscv_net = TimeSeriesSplit(n_splits=k)

# Store evaluation metrics
mse_scores_net = []
rmse_scores_net = []

# Train and evaluate the model using cross-validation
for train_index, test_index in tscv_net.split(X_net):
    
    X_train_net, X_test_net = X_net.iloc[train_index], X_net.iloc[test_index]
    y_train_net, y_test_net = y_net[train_index], y_net[test_index]

    # Apply the date range filter to both X_train_net and y_train_net
    common_dates = (X_train_net.index >= start_date) & (X_train_net.index <= end_date)
    X_train_net = X_train_net[common_dates]
    y_train_net = y_train_net[common_dates]

    # Fit the k-NN model
    knn_net.fit(X_train_net[all_features_net], y_train_net)

    # Forecast
    forecast_features_net = X_test_net[all_features_net].iloc[-24:]
    forecasted_load_net = knn_net.predict(forecast_features_net)

    # Evaluate the model
    mse_net = mean_squared_error(y_test_net[-24:], forecasted_load_net)
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

hourly_rmse_net = []
hourly_mse_net = []

# Initialize lists to store RMSE and MSE per weekday
rmse_per_weekday_net = [0] * 7
mse_per_weekday_net = [0] * 7
count_per_weekday_net = [0] * 7

for month in [7, 8]:  # July and August of 2013
    for day in range(1, 32):
        for hour in range(24):  
            prediction_datetime = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")

            # prediction day is after the end_date for training
            if prediction_datetime > end_date or (hour >= 12 and hour < 24 and day > 1):
                # Select relevant features for the prediction time
                forecast_features_net = X_net[all_features_net].loc[prediction_datetime]

                # Make the hour-ahead forecast for net load using k-NN (previously trained)
                forecasted_net_load = knn_net.predict(forecast_features_net.values.reshape(1, -1))

                # Extract the observed values for the same time from net_load
                observed_value_net = net_load.loc[prediction_datetime]

                # Calculate RMSE for the current hour for net load
                rmse_net = np.sqrt(mean_squared_error(observed_value_net, forecasted_net_load))
                hourly_rmse_net.append(rmse_net)

                # Calculate MSE for the current hour for net load
                mse_net = mean_squared_error(observed_value_net, forecasted_net_load)
                hourly_mse_net.append(mse_net)

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

# Calculate monthly RMSE for net load by taking the average for each month
july_rmse_net = np.mean(hourly_rmse_net[:31 * 24])  # RMSE for July for net load
august_rmse_net = np.mean(hourly_rmse_net[31 * 24:])  # RMSE for August for net load
july_mse_net = np.mean(hourly_mse_net[:31 * 24])  # MSE for July for net load
august_mse_net = np.mean(hourly_mse_net[31 * 24:])  # MSE for August for net load

print("Average RMSE for July (Net Load):", july_rmse_net)
print("Average RMSE for August (Net Load):", august_rmse_net)
print("Average MSE for July (Net Load):", july_mse_net)
print("Average MSE for August (Net Load):", august_mse_net)


