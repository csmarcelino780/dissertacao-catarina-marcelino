# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:36:05 2024

@author: catar
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np




SARIMA_V25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\SARIMA\PV25\SARIMA_scores_verao_load.xlsx")
SARIMA_I25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\SARIMA\PV25\SARIMA_scores_inverno_load.xlsx")
SARIMA_V50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\SARIMA\PV50\SARIMA_scores_verao_load.xlsx")
SARIMA_I50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\SARIMA\PV50\SARIMA_scores_inverno_load.xlsx")


KNN_V25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\knn\pv25\knn_scores_verao_load.xlsx")
KNN_I25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\knn\pv25\knn_scores_inverno_load.xlsx")
KNN_V50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\knn\pv50\knn_scores_verao_load.xlsx")
KNN_I50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\knn\pv50\knn_scores_inverno_load.xlsx")


MLR_V25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Regressao multipla variada\pv25\MLR_scores_verao_load.xlsx")
MLR_I25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Regressao multipla variada\pv25\MLR_scores_inverno_load.xlsx")
MLR_V50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Regressao multipla variada\pv50\MLR_scores_verao_load.xlsx")
MLR_I50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Regressao multipla variada\pv50\MLR_scores_inverno_load.xlsx")


ANN_V25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\ANN\pv25\ANN_scores_verao_load.xlsx")
ANN_I25 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\ANN\pv25\ANN_scores_inverno_load.xlsx")
ANN_V50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\ANN\pv50\ANN_scores_verao_load.xlsx")
ANN_I50 = pd.ExcelFile(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\ANN\pv50\ANN_scores_inverno_load.xlsx")

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def calculate_metrics(real_values, forecasted_values):

    real_values['DateTime'] = pd.to_datetime(real_values['DateTime'])
    forecasted_values['DateTime'] = pd.to_datetime(forecasted_values['DateTime'])

    df = pd.merge(real_values, forecasted_values, on='DateTime', how='inner', suffixes=('_real', '_forecasted'))

    metrics_list = []

    for index, row in df.iterrows():
        rmse = np.sqrt(np.mean((row['Real_Consumption'] - row['Forecasted_Values'])**2))
        mae = np.mean(np.abs(row['Real_Consumption'] - row['Forecasted_Values']))
        mape = np.mean(np.abs((row['Real_Consumption'] - row['Forecasted_Values']) / row['Real_Consumption'])) * 100

        metrics_list.append({
            'DateTime': row['DateTime'],
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })


    metrics_df = pd.DataFrame(metrics_list)

    return metrics_df


def calculate_hourly_metrics(real_values, forecasted_values):

    real_values['DateTime'] = pd.to_datetime(real_values['DateTime'])
    forecasted_values['DateTime'] = pd.to_datetime(forecasted_values['DateTime'])


    df = pd.merge(real_values, forecasted_values, on='DateTime', how='inner', suffixes=('_real', '_forecasted'))


    hourly_metrics_list = []

    for hour in range(24):
        hour_mask = (df['DateTime'].dt.hour == hour)
        hour_data = df[hour_mask]

        rmse = np.sqrt(np.mean((hour_data['Real_Consumption'] - hour_data['Forecasted_Values'])**2))
        mae = np.mean(np.abs(hour_data['Real_Consumption'] - hour_data['Forecasted_Values']))
        mape = np.mean(np.abs((hour_data['Real_Consumption'] - hour_data['Forecasted_Values']) / hour_data['Real_Consumption'])) * 100

        hourly_metrics_list.append({
            'Hour': hour,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })


    hourly_metrics_df = pd.DataFrame(hourly_metrics_list)

    return hourly_metrics_df

def calculate_weekday_metrics(real_values, forecasted_values):

    real_values['DateTime'] = pd.to_datetime(real_values['DateTime'])
    forecasted_values['DateTime'] = pd.to_datetime(forecasted_values['DateTime'])


    df = pd.merge(real_values, forecasted_values, on='DateTime', how='inner', suffixes=('_real', '_forecasted'))


    weekday_metrics_list = []

    for weekday in range(7):
        weekday_mask = (df['DateTime'].dt.weekday == weekday)
        weekday_data = df[weekday_mask]

        hourly_metrics = []

        for hour in range(24):
            hour_mask = (weekday_data['DateTime'].dt.hour == hour)
            hour_data = weekday_data[hour_mask]

            rmse = np.sqrt(np.mean((hour_data['Real_Consumption'] - hour_data['Forecasted_Values'])**2))
            mae = np.mean(np.abs(hour_data['Real_Consumption'] - hour_data['Forecasted_Values']))
            mape = np.mean(np.abs((hour_data['Real_Consumption'] - hour_data['Forecasted_Values']) / hour_data['Real_Consumption'])) * 100

            hourly_metrics.append({
                'Hour': hour,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            })


        hourly_metrics_df = pd.DataFrame(hourly_metrics)
        hourly_metrics_df['Weekday'] = weekday  


        weekday_metrics_list.append(hourly_metrics_df)


    weekday_metrics_df = pd.concat(weekday_metrics_list)

    return weekday_metrics_df

def calculate_monthly_average(metrics_df):
 
    metrics_df['DateTime'] = pd.to_datetime(metrics_df['DateTime'])


    metrics_df['Month'] = metrics_df['DateTime'].dt.month
    metrics_df['Year'] = metrics_df['DateTime'].dt.year

    monthly_average_df = metrics_df.groupby(['Year', 'Month']).mean().reset_index()

    return monthly_average_df

def calculate_overall_average(metrics_df):

    overall_average = metrics_df.mean().to_frame().T

    return overall_average

# List of Excel files
excel_files = [
    "SARIMA_V25", "SARIMA_I25", "SARIMA_V50", "SARIMA_I50",
    "KNN_V25", "KNN_I25", "KNN_V50", "KNN_I50",
    "MLR_V25", "MLR_I25", "MLR_V50", "MLR_I50",
    "ANN_V25", "ANN_I25", "ANN_V50", "ANN_I50"
]





def plot_and_save_rmse_heatmap_side_by_side(data1, data2, filename1, filename2, title1, title2):
    fig, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, figsize=(25, 8), gridspec_kw={'width_ratios': [1, 1, 0.05]})

    
    sns.set(font_scale=1.5)  

    # Choose color 
    cmap = sns.color_palette("RdBu_r", as_cmap=True) 

    # Remove grid lines
    sns.set_style("white")

    # Plot for the first dataset
    heatmap1 = sns.heatmap(data1, cmap=cmap, annot=False,
                           linewidths=0.5, linecolor='black',
                           yticklabels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                           ax=ax1, cbar_ax=cbar_ax , vmin=0, vmax=45)
    heatmap1.set_xlabel("Hour of Day", fontsize=16)
    heatmap1.set_ylabel("Day of Week", fontsize=16)
    ax1.set_title(title1, fontsize=18)

    # Rotate y-axis labels to horizontal
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    # Plot for the second dataset
    heatmap2 = sns.heatmap(data2, cmap=cmap, annot=False,
                           linewidths=0.5, linecolor='black',
                           yticklabels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                           ax=ax2, cbar_ax=cbar_ax, vmin=0, vmax=45)
    heatmap2.set_xlabel("Hour of Day", fontsize=16)
    heatmap2.set_ylabel("Day of Week", fontsize=16)
    ax2.set_title(title2, fontsize=18)

    # Rotate y-axis labels to horizontal
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    # Provide a color bar title
    cbar = heatmap1.collections[0].colorbar
    cbar.set_label("RMSE (kWh)", fontsize=16)

   
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    
    plt.savefig(filename1)
    plt.savefig(filename2)
    plt.close()



with pd.ExcelWriter('Metrics_metho1_with_heatmap_load.xlsx') as writer:

    for i in range(0, len(excel_files), 2):
        file1 = excel_files[i]
        file2 = excel_files[i + 1]

        
        real_values_file1 = globals()[file1].parse('Real_Values')
        forecasted_values_file1 = globals()[file1].parse('Method1')

        
        real_values_file2 = globals()[file2].parse('Real_Values')
        forecasted_values_file2 = globals()[file2].parse('Method1')

        # Calculate metrics for each data point for the first pair
        metrics_df1 = calculate_metrics(real_values_file1, forecasted_values_file1)

        # Print or save the results for the first pair
        print(f"Metrics for {file1}:")
        print(metrics_df1)

        
        metrics_df1.to_excel(writer, sheet_name=f'{file1}_all_points_metrics', index=False)

        
        hourly_metrics_df1 = calculate_hourly_metrics(real_values_file1, forecasted_values_file1)
        hourly_metrics_df1.to_excel(writer, sheet_name=f'{file1}_hourly_metrics', index=False)

        
        weekday_metrics_df1 = calculate_weekday_metrics(real_values_file1, forecasted_values_file1)

        
        rmse_heatmap_data1 = weekday_metrics_df1.pivot_table(index='Weekday', columns='Hour', values='RMSE')

        
        metrics_df2 = calculate_metrics(real_values_file2, forecasted_values_file2)

        # Print or save the results for the second pair
        print(f"Metrics for {file2}:")
        print(metrics_df2)

        
        metrics_df2.to_excel(writer, sheet_name=f'{file2}_all_points_metrics', index=False)

        
        hourly_metrics_df2 = calculate_hourly_metrics(real_values_file2, forecasted_values_file2)
        hourly_metrics_df2.to_excel(writer, sheet_name=f'{file2}_hourly_metrics', index=False)

        
        weekday_metrics_df2 = calculate_weekday_metrics(real_values_file2, forecasted_values_file2)

        
        rmse_heatmap_data2 = weekday_metrics_df2.pivot_table(index='Weekday', columns='Hour', values='RMSE')

        
        plot_and_save_rmse_heatmap_side_by_side(rmse_heatmap_data1, rmse_heatmap_data2,
                                                f'{file1}_weekday_rmse_heatmap_load.png', f'{file2}_weekday_rmse_heatmap_load.png',
                                                'A', 'B')

       
        weekday_metrics_df1.to_excel(writer, sheet_name=f'{file1}_weekday_metrics', index=False)

        
        monthly_average_df1 = calculate_monthly_average(metrics_df1)
        monthly_average_df1.to_excel(writer, sheet_name=f'{file1}_monthly_average', index=False)

        
        weekday_metrics_df2.to_excel(writer, sheet_name=f'{file2}_weekday_metrics', index=False)

        
        monthly_average_df2 = calculate_monthly_average(metrics_df2)
        monthly_average_df2.to_excel(writer, sheet_name=f'{file2}_monthly_average', index=False)

