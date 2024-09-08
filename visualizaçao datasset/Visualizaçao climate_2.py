# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:34:44 2024

@author: catar
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read the data from the Excel file into a DataFrame
data = pd.read_excel(r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Clustering de dados climaticos\best_cluster_data.xlsx")

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Assuming the date format is in 'yyyy-mm-dd', you can filter data for different weeks
weeks = [
    (pd.to_datetime('2013-07-15'), pd.to_datetime('2013-07-22')),  # Week 1 - July
    (pd.to_datetime('2013-11-18'), pd.to_datetime('2013-11-24')),  # Week 2 - November
    # Add more weeks as needed
]

# Define titles for each week
titles = ["Plotting Data for a Specific Week in July", "Plotting Data for a Specific Week in November"]  # Add more titles as needed

# Plotting data for each week
for i, (start_date, end_date) in enumerate(weeks, start=1):
    plt.figure(figsize=(15, 7))  # Adjust figure size for side-by-side plots

    # Filter data for the week
    week_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # Plotting first three parameters side by side
    plt.subplot(1, 2, 1)
    for col in week_data.columns[1:4]:  # Selecting first three parameters
        plt.plot(week_data['date'], week_data[col], label=col)
    plt.title(titles[i-1])  # Use the corresponding title for the current week
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a'))

    # Plotting last three parameters side by side
    plt.subplot(1, 2, 2)
    for col in week_data.columns[4:]:  # Selecting last three parameters
        plt.plot(week_data['date'], week_data[col], label=col)
    plt.title(titles[i-1])  # Use the corresponding title for the current week
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a'))

    plt.tight_layout()

plt.show()
