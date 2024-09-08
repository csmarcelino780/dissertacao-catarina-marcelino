# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:47:20 2023

@author: catar
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import holidays
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
import holidays
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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



#################### ANN Structure #################################
def load_forecast_ann(X_train, y_train, X_test, y_test, all_features):
    # Initialize a Sequential model
    ann_model = Sequential()

    # Add layers to the model
    ann_model.add(Dense(units=10, activation='relu', input_dim=len(all_features)))  # Adjust the number of units as needed
    ann_model.add(Dropout(0.2))  # Optional dropout layer for regularization
    ann_model.add(Dense(units=10, activation='relu'))
    ann_model.add(Dense(units=1, activation='linear'))  # Output layer with linear activation for regression

    # Compile the model
    ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  # Use an appropriate optimizer and loss function

    # Train the ANN model
    ann_model.fit(X_train[all_features], y_train, epochs=10, batch_size=100, validation_data=(X_test[all_features], y_test))

    # Forecast using the trained ANN model
    forecasted_load = ann_model.predict(X_test[all_features])

    return forecasted_load

# Load and preprocess real data
data_real = load_real_data(data_df)
data_real_resampled = resample_real_data(data_real)
data = load_and_preprocess_data(data_df)
data_resampled = resample_and_filter_data(data)

# Prepare data for training and evaluation
X = data_resampled.iloc[:-24].dropna()  # Exclude the last 24 rows + dropping rows with NaN values
valid_indices = X.index  # Store the indices of the non-NaN rows in X

y = data_resampled['MT_080_normalized'].iloc[24:].values  # Exclude the first 24 values
valid_positions = [data_resampled.index.get_loc(idx) for idx in valid_indices]  # List of positions in data_resampled for the valid_indices
y = y[valid_positions]  # Keep only the values corresponding to non-NaN rows in X

# Store the best clusters and features
best_clusters = {}
best_features = []

# Empty DataFrame with an index based on the non-NaN rows in X
cluster_values_df = pd.DataFrame(index=X.index)
# Call cluster_features_for_data
cluster_values_df_load = cluster_features_for_data(clima_file, X, k)
# Merge cluster_values_df with 'X' using a common index
X = X.join(cluster_values_df)
X.dropna(inplace=True)  # Drop NaN rows in X
valid_indices = X.index  # Update the valid_indices

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

#################### ANN Structure #################################
ann_model = Sequential()
ann_model.add(Dense(units=10, activation='relu', input_dim=len(all_features)))
ann_model.add(Dropout(0.2))
ann_model.add(Dense(units=10, activation='relu'))
ann_model.add(Dense(units=1, activation='linear'))
ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

################################################################################
# TimeSeriesSplit for cross-validation
tscv_ann = TimeSeriesSplit(n_splits=k)

# Store evaluation metrics
mse_scores_ann = []
rmse_scores_ann = []

# Train and evaluate the model using cross-validation
for train_index_ann, test_index_ann in tscv_ann.split(X):
    
    X_train_ann, X_test_ann = X.iloc[train_index_ann], X.iloc[test_index_ann]
    y_train_ann, y_test_ann = y[train_index_ann], y[test_index_ann]

    # Apply the date range filter to both X_train_ann and y_train_ann
    common_dates_ann = (X_train_ann.index >= start_date) & (X_train_ann.index <= end_date)
    X_train_ann = X_train_ann[common_dates_ann]
    y_train_ann = y_train_ann[common_dates_ann]

    # Train the ANN model
    ann_model.fit(X_train_ann[all_features], y_train_ann, epochs=10, batch_size=100, validation_data=(X_test_ann[all_features], y_test_ann))

    # Forecast using the trained ANN model
    forecasted_load_ann = ann_model.predict(X_test_ann[all_features])

    # Evaluate the ANN model
    mse_ann = mean_squared_error(y_test_ann, forecasted_load_ann[-len(y_test_ann):])
    mse_scores_ann.append(mse_ann)

    rmse_ann = np.sqrt(mse_ann)
    rmse_scores_ann.append(rmse_ann)

# Calculate average MSE and RMSE using ANN
avg_mse_ann = np.mean(mse_scores_ann)
avg_rmse_ann = np.mean(rmse_scores_ann)

# Output the evaluation metrics using ANN
print(f"Average MSE (ANN): {avg_mse_ann}")
print(f"Average RMSE (ANN): {avg_rmse_ann}")

# Months to forecast
months = [7, 8]  # July and August
days_in_month = [31, 31]  # Number of days in July and August, adjust as needed

monthly_rmse_ann = {month: [] for month in months}
monthly_mse_ann = {month: [] for month in months}

forecasted_loads_ann = []

# Convert end_date to a Timestamp
end_date = pd.to_datetime(end_date)

hourly_rmse_ann = []
hourly_mse_ann = []  

# Make day-ahead forecasts for all the hours of each day using ANN
for month in [7, 8]:  # July and August of 2013
    for day in range(1, 32):
        for hour in range(24):
            prediction_datetime = pd.to_datetime(f"2013-{month:02d}-{day:02d} {hour:02d}:00:00")
            
            # Prediction day is after the end_date for training
            if prediction_datetime > end_date or (hour >= 12 and hour < 24 and day > 1):
                # Select relevant features for the prediction time
                forecast_features = X[all_features].loc[prediction_datetime]
                
                # Make the hour-ahead forecast using the trained ANN model
                forecasted_load_ann = load_forecast_ann(X_train_ann, y_train_ann, X_test_ann, y_test_ann, all_features)
                
                # Append the forecasted load to the results
                forecasted_loads_ann.append((prediction_datetime, forecasted_load_ann[0]))

                # Evaluate the ANN model
                observed_values_ann = y_test_ann[-len(forecasted_load_ann):]  # Adjusted to match lengths
                
                # Calculate RMSE for the current hour
                rmse_ann = np.sqrt(mean_squared_error(observed_values_ann, forecasted_load_ann))
                hourly_rmse_ann.append(rmse_ann)
                
                # Calculate MSE for the current hour
                mse_ann = mean_squared_error(observed_values_ann, forecasted_load_ann)
                hourly_mse_ann.append(mse_ann)

# Forecasts to a DataFrame
forecast_df_ann = pd.DataFrame(forecasted_loads_ann, columns=["DateTime", "Forecasted_Load_ANN"])
forecast_df_ann.set_index("DateTime", inplace=True)

# Calculate monthly RMSE by taking the average for each month using ANN
july_rmse_ann = np.mean(hourly_rmse_ann[:31 * 24])  # RMSE for July using ANN
august_rmse_ann = np.mean(hourly_rmse_ann[31 * 24:])  # RMSE for August using ANN
july_mse_ann = np.mean(hourly_mse_ann[:31 * 24])  # MSE for July using ANN
august_mse_ann = np.mean(hourly_mse_ann[31 * 24:])  # MSE for August using ANN

print("Average RMSE for July (ANN):", july_rmse_ann)
print("Average RMSE for August (ANN):", august_rmse_ann)

