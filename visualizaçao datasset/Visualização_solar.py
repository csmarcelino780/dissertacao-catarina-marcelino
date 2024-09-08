# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:50:44 2023

@author: catar
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


data = pd.read_csv(r'C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Dados\programas\PVproduction_modified_2.csv', parse_dates=['Time'],usecols=["PV25","PV50","Time"])
# # Convert to datetime data type
data['Time'] = pd.to_datetime(data['Time'])

###############ONE YEAR ##################
# day_array=data["DateTime"].values 
# net_load_M80_array=data["MT_080"].values
# plt.figure(figsize=(20,8)) # plot size 
# plt.plot(day_array , net_load_M80_array)

# ##############ONE month  ##################

# # Filter the data for a specific month
# desired_month = pd.to_datetime('2012-08-01')  # pick the month to plot
# filtered_data = data[data['DateTime'].dt.to_period('M') == desired_month.to_period('M')] # filter the data for the selected month

# # Plot the filtered data
# plt.figure(figsize=(16,8))
# plt.plot(filtered_data['DateTime'], filtered_data['MT_080'])

# # Add labels and a title
# plt.xlabel('Datetime')
# plt.ylabel('load')
# plt.title('Plotting Data for a Specific Month')

# # Display the plot
# plt.show()

################ONE WEEK#####################
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# Define a function to convert a date to an abbreviated day name
def date_to_day(date):
    return date.strftime('%a')

# July
start_date_july = pd.to_datetime('2013-07-15')
end_date_july = start_date_july + pd.DateOffset(weeks=1)
filtered_data_july = data[(data['Time'] >= start_date_july) & (data['Time'] < end_date_july)]

# November
start_date_november = pd.to_datetime('2013-11-18')
end_date_november = start_date_november + pd.DateOffset(weeks=1)
filtered_data_november = data[(data['Time'] >= start_date_november) & (data['Time'] < end_date_november)]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# Plot for July
axs[0].plot(filtered_data_july['Time'], filtered_data_july['PV25'], label='PV25 - 10.7 kW')
axs[0].plot(filtered_data_july['Time'], filtered_data_july['PV50'], label='PV50 - 21.4 kW')
axs[0].set_ylim(0, 18)
axs[0].set_xlabel('Datetime',fontsize=17)
axs[0].set_ylabel("Solar Production (kW)",fontsize=17)
axs[0].set_title('Plotting Data for a Specific Week - July')
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%a'))
axs[0].tick_params(axis='both', which='major', labelsize=18)  # Increase the font size of the tick labels

# Plot for November
axs[1].plot(filtered_data_november['Time'], filtered_data_november['PV25'], label='PV25 - 10.7 kW')
axs[1].plot(filtered_data_november['Time'], filtered_data_november['PV50'], label='PV50 - 21.4 kW')
axs[1].set_ylim(0, 18)
axs[1].set_xlabel('Datetime',fontsize=17)
axs[1].set_ylabel('Solar Production (kW)',fontsize=17)
axs[1].set_title('Plotting Data for a Specific Week - November')
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%a'))
axs[1].tick_params(axis='both', which='major', labelsize= 18)  # Increase the font size of the tick labels
# Common title for both subplots
plt.suptitle('Comparison of Weekly Data for scenario 1  - July and November', fontsize=16)

# Add legend
for ax in axs:
    ax.legend()

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rectangle to make space for the common title

# Display the plots
plt.show()

# #######################ONE DAY####################

# # Filter the data for a specific day
# desired_date = pd.to_datetime('2012-05-01')  # pick the day to plot
# filtered_data = data[data['DateTime'].dt.date == desired_date.date()] #filter the data from the day picked

# # Plot the filtered data
# plt.figure(figsize=(16,8)) # plot size 
# plt.plot(filtered_data['DateTime'],filtered_data['MT_080'] )

# # Add labels and a title
# plt.xlabel('Datetime')
# plt.ylabel('load')
# plt.title('Plotting Data for a Specific Day')

# # Display the plot
# plt.show()

############################## ONE HOUR##############
 
# desired_hour = pd.to_datetime('2012-01-15 23:30:00')   #pick hour to plot 
# filtered_data = data[data['DateTime'].dt.floor('H') == desired_hour.floor('H')] # filter the data form the hour chosen 

# # Plot the filtered data
# plt.figure(figsize=(16,8))
# plt.plot(filtered_data['DateTime'], filtered_data['MT_080'])

# # Add labels and a title
# plt.xlabel('Datetime')
# plt.ylabel('load')
# plt.title('Plotting Data for a Specific Hour')

# # Display the plot
# plt.show()




