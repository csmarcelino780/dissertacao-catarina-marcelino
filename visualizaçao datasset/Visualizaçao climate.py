# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:47:24 2024

@author: catar
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read the data from the Excel file into a DataFrame
data = pd.read_excel(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Clustering de dados climaticos\best_cluster_data.xlsx")

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Assuming the date format is in 'yyyy-mm-dd', you can filter data for two different weeks
# For example, let's say you want to plot the data for the first week and the second week
start_date_week1 = pd.to_datetime('2013-07-15')
end_date_week1 = pd.to_datetime('2013-07-22')

start_date_week2 = pd.to_datetime('2013-11-18')
end_date_week2 = pd.to_datetime('2013-11-24')

# Filter data for the first week
week1_data = data[(data['date'] >= start_date_week1) & (data['date'] <= end_date_week1)]

# Filter data for the second week
week2_data = data[(data['date'] >= start_date_week2) & (data['date'] <= end_date_week2)]

# Plotting data for the first week
plt.figure(figsize=(15, 10))

# Plotting first three parameters for the first week
plt.subplot(2, 2, 1)
for col in week1_data.columns[1:4]:  # Selecting first three parameters
    plt.plot(week1_data['date'], week1_data[col], label=col)
plt.title('Plotting First Three Parameters for Week - July')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

# Plotting last three parameters for the first week
plt.subplot(2, 2, 2)
for col in week1_data.columns[4:]:  # Selecting last three parameters
    plt.plot(week1_data['date'], week1_data[col], label=col)
plt.title('Plotting Last Three Parameters for Week - July')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

# Plotting data for the second week
plt.subplot(2, 2, 3)

# Plotting first three parameters for the second week
for col in week2_data.columns[1:4]:  # Selecting first three parameters
    plt.plot(week2_data['date'], week2_data[col], label=col)
plt.title('Plotting First Three Parameters for Week - November')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

# Plotting last three parameters for the second week
plt.subplot(2, 2, 4)
for col in week2_data.columns[4:]:  # Selecting last three parameters
    plt.plot(week2_data['date'], week2_data[col], label=col)
plt.title('Plotting Last Three Parameters for Week - November')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

# Replace x-axis labels with day of the week
plt.xticks(plt.xticks()[0], [pd.to_datetime(x).strftime('%a') for x in plt.xticks()[0]])

plt.tight_layout()
plt.show()
