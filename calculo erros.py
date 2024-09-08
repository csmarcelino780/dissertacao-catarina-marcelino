# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:19:59 2023

@author: catar
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


# Load the Excel file
SARIMA_V25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\SARIMA\PV25\SARIMA_scores_verao.xlsx")
SARIMA_I25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\SARIMA\PV25\SARIMA_scores_inverno.xlsx")
SARIMA_V50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\SARIMA\PV50\SARIMA_scores_verao.xlsx")
SARIMA_I50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\SARIMA\PV50\SARIMA_scores_inverno.xlsx")

# Load the Excel file
KNN_V25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\knn\pv25\knn_scores_verao.xlsx")
KNN_I25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\knn\pv25\knn_scores_inverno.xlsx")
KNN_V50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\knn\pv50\knn_scores_verao.xlsx")
KNN_I50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\knn\pv50\knn_scores_inverno.xlsx")

# Load the Excel file
MLR_V25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Regressao multipla variada\pv25\MLR_scores_verao.xlsx")
MLR_I25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Regressao multipla variada\pv25\MLR_scores_inverno.xlsx")
MLR_V50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Regressao multipla variada\pv50\MLR_scores_verao.xlsx")
MLR_I50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Regressao multipla variada\pv50\MLR_scores_inverno.xlsx")

# Load the Excel file
ANN_V25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\ANN\pv25\ANN_scores_verao.xlsx")
ANN_I25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\ANN\pv25\ANN_scores_inverno.xlsx")
ANN_V50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\ANN\pv50\ANN_scores_verao.xlsx")
ANN_I50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\ANN\pv50\ANN_scores_inverno.xlsx")


import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def calculate_metrics(real_values, forecasted_values):
    # Convert 'DateTime' to datetime format if it's not already
    real_values['DateTime'] = pd.to_datetime(real_values['DateTime'])
    forecasted_values['DateTime'] = pd.to_datetime(forecasted_values['DateTime'])

    # Merge real and forecasted values on 'DateTime'
    df = pd.merge(real_values, forecasted_values, on='DateTime', how='inner', suffixes=('_real', '_forecasted'))

    # Calculate metrics for each data point
    metrics_list = []

    for index, row in df.iterrows():
        rmse = np.sqrt(np.mean((row['Real_Net_Consumption'] - row['Forecasted_Values'])**2))
        mae = np.mean(np.abs(row['Real_Net_Consumption'] - row['Forecasted_Values']))
        mape = np.mean(np.abs((row['Real_Net_Consumption'] - row['Forecasted_Values']) / row['Real_Net_Consumption'])) * 100

        metrics_list.append({
            'DateTime': row['DateTime'],
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })

    # Convert the list of dictionaries to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    return metrics_df


def calculate_hourly_metrics(real_values, forecasted_values):
    # Convert 'DateTime' to datetime format if it's not already
    real_values['DateTime'] = pd.to_datetime(real_values['DateTime'])
    forecasted_values['DateTime'] = pd.to_datetime(forecasted_values['DateTime'])

    # Merge real and forecasted values on 'DateTime'
    df = pd.merge(real_values, forecasted_values, on='DateTime', how='inner', suffixes=('_real', '_forecasted'))

    # Calculate metrics for each hour
    hourly_metrics_list = []

    for hour in range(24):
        hour_mask = (df['DateTime'].dt.hour == hour)
        hour_data = df[hour_mask]

        rmse = np.sqrt(np.mean((hour_data['Real_Net_Consumption'] - hour_data['Forecasted_Values'])**2))
        mae = np.mean(np.abs(hour_data['Real_Net_Consumption'] - hour_data['Forecasted_Values']))
        mape = np.mean(np.abs((hour_data['Real_Net_Consumption'] - hour_data['Forecasted_Values']) / hour_data['Real_Net_Consumption'])) * 100

        hourly_metrics_list.append({
            'Hour': hour,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })

    # Convert the list of dictionaries to a DataFrame
    hourly_metrics_df = pd.DataFrame(hourly_metrics_list)

    return hourly_metrics_df

def calculate_weekday_metrics(real_values, forecasted_values):
    # Convert 'DateTime' to datetime format if it's not already
    real_values['DateTime'] = pd.to_datetime(real_values['DateTime'])
    forecasted_values['DateTime'] = pd.to_datetime(forecasted_values['DateTime'])

    # Merge real and forecasted values on 'DateTime'
    df = pd.merge(real_values, forecasted_values, on='DateTime', how='inner', suffixes=('_real', '_forecasted'))

    # Calculate metrics for each weekday and hour
    weekday_metrics_list = []

    for weekday in range(7):
        weekday_mask = (df['DateTime'].dt.weekday == weekday)
        weekday_data = df[weekday_mask]

        hourly_metrics = []

        for hour in range(24):
            hour_mask = (weekday_data['DateTime'].dt.hour == hour)
            hour_data = weekday_data[hour_mask]

            rmse = np.sqrt(np.mean((hour_data['Real_Net_Consumption'] - hour_data['Forecasted_Values'])**2))
            mae = np.mean(np.abs(hour_data['Real_Net_Consumption'] - hour_data['Forecasted_Values']))
            mape = np.mean(np.abs((hour_data['Real_Net_Consumption'] - hour_data['Forecasted_Values']) / hour_data['Real_Net_Consumption'])) * 100

            hourly_metrics.append({
                'Hour': hour,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            })

        # Convert the list of dictionaries to a DataFrame for each hour
        hourly_metrics_df = pd.DataFrame(hourly_metrics)
        hourly_metrics_df['Weekday'] = weekday  # Add weekday column to the DataFrame

        # Append the hourly_metrics_df to the list
        weekday_metrics_list.append(hourly_metrics_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    weekday_metrics_df = pd.concat(weekday_metrics_list)

    return weekday_metrics_df

def calculate_monthly_average(metrics_df):
    # Convert 'DateTime' to datetime format if it's not already
    metrics_df['DateTime'] = pd.to_datetime(metrics_df['DateTime'])

    # Extract month and year from the 'DateTime' column
    metrics_df['Month'] = metrics_df['DateTime'].dt.month
    metrics_df['Year'] = metrics_df['DateTime'].dt.year

    # Calculate monthly averages
    monthly_average_df = metrics_df.groupby(['Year', 'Month']).mean().reset_index()

    return monthly_average_df

def calculate_overall_average(metrics_df):
    # Calculate overall average
    overall_average = metrics_df.mean().to_frame().T

    return overall_average

# List of Excel files
excel_files = [
    "SARIMA_V25", "SARIMA_I25", "SARIMA_V50", "SARIMA_I50",
    "KNN_V25", "KNN_I25", "KNN_V50", "KNN_I50",
    "MLR_V25", "MLR_I25", "MLR_V50", "MLR_I50",
    "ANN_V25", "ANN_I25", "ANN_V50", "ANN_I50"
]



import seaborn as sns

def plot_and_save_rmse_heatmap(data, filename):
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.2)  # Adjust font size
    cmap = sns.color_palette("YlGnBu", as_cmap=True)  # Choose color palette

    # Custom style parameters
    sns.set_style("whitegrid", {'axes.edgecolor': '0.8'})
    sns.set_palette("viridis", n_colors=10)

    heatmap = sns.heatmap(data, cmap=cmap, annot=True, fmt=".2f", cbar_kws={'label': 'RMSE'},
                          linewidths=0.5, linecolor='black', annot_kws={"size": 10},
                          yticklabels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Set x-axis and y-axis labels
    heatmap.set_xlabel("Hour of Day", fontsize=14)
    heatmap.set_ylabel("Day of Week", fontsize=14)

    plt.title("RMSE Heatmap - Day of Week vs. Hour of Day", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Create an Excel writer
with pd.ExcelWriter('Metrics_metho2_with_heatmap.xlsx') as writer:
    # Iterate over each Excel file
    for file in excel_files:
        # Load the Excel files
        real_values_file = globals()[file].parse('Real_Values')
        forecasted_values_file = globals()[file].parse('Method2')

        # Calculate metrics for each data point
        metrics_df = calculate_metrics(real_values_file, forecasted_values_file)

        # Print or save the results
        print(f"Metrics for {file}:")
        print(metrics_df)

        # Save the results in a separate sheet
        metrics_df.to_excel(writer, sheet_name=f'{file}_all_points_metrics', index=False)

        # Calculate and save hourly metrics
        hourly_metrics_df = calculate_hourly_metrics(real_values_file, forecasted_values_file)
        hourly_metrics_df.to_excel(writer, sheet_name=f'{file}_hourly_metrics', index=False)

         # Calculate and save weekday metrics
        weekday_metrics_df = calculate_weekday_metrics(real_values_file, forecasted_values_file)

        # Create a pivot table for the RMSE heatmap
        rmse_heatmap_data = weekday_metrics_df.pivot_table(index='Weekday', columns='Hour', values='RMSE')

        # Save the RMSE heatmap to a separate sheet
        plot_and_save_rmse_heatmap(rmse_heatmap_data, f'{file}_weekday_rmse_heatmap.png')


        # Save the weekday metrics to a separate sheet
        weekday_metrics_df.to_excel(writer, sheet_name=f'{file}_weekday_metrics', index=False)

        # Calculate and save monthly averages
        monthly_average_df = calculate_monthly_average(metrics_df)
        monthly_average_df.to_excel(writer, sheet_name=f'{file}_monthly_average', index=False)
