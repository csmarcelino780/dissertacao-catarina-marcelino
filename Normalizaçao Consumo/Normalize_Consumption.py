# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:17:50 2023

@author: catar
"""
# Import required libraries
import pandas as pd
from scipy.stats import zscore

# Load the DataFrame from the CSV file
load_real_2 = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Demand_Tese_teste.csv', 
                          parse_dates=['DateTime'],
                          usecols=["MT_080", "DateTime"])

# Normalize the MT_080 column using Z-score
load_real_2['MT_080_normalized'] = zscore(load_real_2['MT_080'])

# Write the DataFrame with the normalized column to a new CSV file
load_real_2.to_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Normalizaçao Consumo\Normalized_Demand_Tese_teste.csv', index=False)

# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# # Load the DataFrame from the CSV file
# load_real_2 = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Demand_Tese_teste.csv', 
#                           parse_dates=['DateTime'],
#                           usecols=["MT_080", "DateTime"])

# # Apply Min-Max normalization to the 'MT_080' column
# scaler = MinMaxScaler()
# load_real_2['MT_080_normalized'] = scaler.fit_transform(load_real_2[['MT_080']])

# # Write the DataFrame with the normalized column to a new CSV file
# load_real_2.to_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Normalizaçao Consumo\Normalized_Demand_Tese_teste.csv', index=False)
