# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:14:33 2023

@author: catar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load your consumption data
data_df = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Normalizaçao Consumo\Normalized_Demand_Tese_teste.csv', parse_dates=['DateTime'], usecols=["MT_080_normalized", "DateTime"])

# Filter data based on the specified training date range
start_date = '2012-02-01'
end_date = '2013-06-29'
train_data = data_df[(data_df['DateTime'] >= start_date) & (data_df['DateTime'] <= end_date)]

X_train = train_data[['MT_080_normalized']]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define the range of k values
param_range = list(range(1, 30))

# Calculate RMSE for different values of k
rmse_scores = []
for k in param_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    # Predict cluster labels for each sample
    labels = kmeans.predict(X_train_scaled)
    # Get cluster centers
    centers = kmeans.cluster_centers_
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(X_train_scaled, centers[labels]))
    rmse_scores.append(rmse)

# Plot the elbow plot
plt.figure(figsize=(10, 6))
plt.plot(param_range, rmse_scores, marker='o')
plt.title('Elbow Plot for K-means Clustering (RMSE)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.show()

# Find the optimal k value
optimal_k_rmse = param_range[np.argmin(rmse_scores)]
print(f"The optimal number of clusters (k) based on RMSE is: {optimal_k_rmse}")
