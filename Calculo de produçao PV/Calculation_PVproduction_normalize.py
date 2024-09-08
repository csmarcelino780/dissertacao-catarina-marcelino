# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:11:15 2023

@author: catar
"""
import pandas as pd
from scipy.stats import zscore

# Initialize parameters
start_date = '2013-02-01'
end_date = '2013-08-30'
k = 5

# Scenarios with their respective names and multipliers
scenarios = [
    {"name": "25", "cenario": 0.25},
    {"name": "50", "cenario": 0.5}
]

# Reading and processing PVGIS data
# Your existing code to read the file would be here
data_2 = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\Dados_ProduçaoPV.csv', parse_dates=['Time'], usecols=["Time", "Gb(i)", "Power"])
data_2.set_index('Time', inplace=True)
data_2_resampled = data_2.resample('1H').mean()
data_resampled_2 = data_2_resampled[(data_2_resampled.index >= start_date) & (data_2_resampled.index <= end_date)]

# Normalize the 'Power' column of the resampled data
data_2_resampled['Power_normalized'] = zscore(data_2_resampled['Power'])

# Reading and processing Load Data
# Your existing code to read the file would be here
load_real_2 = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Normalizaçao Consumo\Normalized_Demand_Tese_teste.csv', parse_dates=['DateTime'], usecols=["MT_080_normalized", "DateTime"])
load_real_2.set_index('DateTime', inplace=True)
load_real_2_resampled = load_real_2.resample('1H').mean()
load_real_2_resampled = load_real_2_resampled[load_real_2_resampled.index.date > load_real_2_resampled.index[0].date()]

# Reading additional PV production data
pv_production = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\Dados_ProduçaoPV.csv', parse_dates=['Time'])

# Iterate through scenarios and create a new column for each
for scenario in scenarios:
    cenario = scenario['cenario']
    name = scenario['name']

    max_value = load_real_2_resampled['MT_080_normalized'].max()
    pv_installed = max_value * cenario

    modified_data = data_2_resampled.copy()
    modified_data['Total_PV_Production_{}'.format(name)] = modified_data['Power_normalized'] * pv_installed

    pv_production['pv_installed_{}'.format(name)] = pv_installed
    pv_production['PV{}'.format(name)] = modified_data['Total_PV_Production_{}'.format(name)].values

# Save the modified DataFrame back to a CSV file with normalized values
pv_production.to_csv('PVproduction_modified_normalized_2.csv', index=False)

# Save the modified DataFrame to an Excel file with normalized values
pv_production.to_excel('PVproduction_modified_normalized_3.xlsx', index=False)





# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# # Initialize parameters
# start_date = '2013-02-01'
# end_date = '2013-08-30'
# k = 5

# # Scenarios with their respective names and multipliers
# scenarios = [
#     {"name": "25", "cenario": 0.25},
#     {"name": "50", "cenario": 0.5}
# ]

# # Reading and processing PVGIS data
# # Your existing code to read the file would be here
# data_2 = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\Dados_ProduçaoPV.csv', parse_dates=['Time'], usecols=["Time", "Gb(i)", "Power"])
# data_2.set_index('Time', inplace=True)
# data_2_resampled = data_2.resample('1H').mean()
# data_resampled_2 = data_2_resampled[(data_2_resampled.index >= start_date) & (data_2_resampled.index <= end_date)]

# # Normalize the 'Power' column of the resampled data using Min-Max scaling
# scaler_min_max_power = MinMaxScaler()
# data_2_resampled['Power_normalized'] = scaler_min_max_power.fit_transform(data_2_resampled[['Power']])

# # Reading and processing Load Data
# # Your existing code to read the file would be here
# load_real_2 = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Normalizaçao Consumo\Normalized_Demand_Tese_teste.csv', parse_dates=['DateTime'], usecols=["MT_080_normalized", "DateTime"])
# load_real_2.set_index('DateTime', inplace=True)
# load_real_2_resampled = load_real_2.resample('1H').mean()
# load_real_2_resampled = load_real_2_resampled[load_real_2_resampled.index.date > load_real_2_resampled.index[0].date()]

# # Reading additional PV production data
# pv_production = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Calculo de produçao PV\Dados_ProduçaoPV.csv', parse_dates=['Time'])

# # Iterate through scenarios and create a new column for each
# for scenario in scenarios:
#     cenario = scenario['cenario']
#     name = scenario['name']

#     max_value = load_real_2_resampled['MT_080_normalized'].max()
#     pv_installed = max_value * cenario

#     modified_data = data_2_resampled.copy()
#     modified_data['Total_PV_Production_{}'.format(name)] = modified_data['Power_normalized'] * pv_installed

#     pv_production['pv_installed_{}'.format(name)] = pv_installed
#     pv_production['PV{}'.format(name)] = modified_data['Total_PV_Production_{}'.format(name)].values

# # Save the modified DataFrame back to a CSV file with Min-Max normalized values
# pv_production.to_csv('PVproduction_modified_normalized_2.csv', index=False)

# # Save the modified DataFrame to an Excel file with Min-Max normalized values
# pv_production.to_excel('PVproduction_modified_normalized_3.xlsx', index=False)

