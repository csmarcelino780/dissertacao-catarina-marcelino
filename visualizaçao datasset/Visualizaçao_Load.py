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

# Load the data from Excel
data = pd.read_csv('C:/Users/catar/Desktop/dados tese/Demand_Tese_teste.csv' ,  parse_dates=['Day'], usecols=[ "Day", "Hour" , "MT_080" ,"DateTime"]) 
# Convert to datetime data type
data['DateTime'] = pd.to_datetime(data['DateTime'])



##############ONE YEAR ##################
# day_array=data["DateTime"].values 
# net_load_M80_array=data["MT_080"].values
# plt.figure(figsize=(20,8)) # plot size 
# plt.plot(day_array , net_load_M80_array)



############################################ one week #####################################

# Define a function to convert a date to an abbreviated day name
def date_to_day(date):
    return date.strftime('%a')

# Define the start dates for the two weeks
start_date_july = pd.to_datetime('2013-07-15')
start_date_november = pd.to_datetime('2013-11-18')

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

######################july
# Define week
end_date_july = start_date_july + pd.DateOffset(weeks=1)  # end date is one week after the start date
# Filter the data for the week, only contain the values of MT_080 for the dates chosen
filtered_data_july = data[(data['DateTime'] >= start_date_july) & (data['DateTime'] < end_date_july)]
# Plot the filtered data for July
axs[0].plot(filtered_data_july['DateTime'], filtered_data_july['MT_080'])
# Add labels and a title for July
axs[0].set_ylim(0, 40)
axs[0].set_xlabel('Datetime',fontsize=17)
axs[0].set_ylabel('Load (kW)',fontsize=17)
axs[0].set_title('Plotting Data for a Specific Week in JULY')
# Format the x-axis labels with abbreviated day names
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%a'))
axs[0].tick_params(axis='both', which='major', labelsize=18) 
######################November
# Define week
end_date_november = start_date_november + pd.DateOffset(weeks=1)  # end date is one week after the start date
# Filter the data for the week, only contain the values of MT_080 for the dates chosen
filtered_data_november = data[(data['DateTime'] >= start_date_november) & (data['DateTime'] < end_date_november)]
# Plot the filtered data for November
axs[1].plot(filtered_data_november['DateTime'], filtered_data_november['MT_080'])
# Add labels and a title for November
axs[1].set_ylim(0, 40)
axs[1].set_xlabel('Datetime',fontsize=17)
axs[1].set_ylabel('Load (kW)',fontsize=17)
axs[1].set_title('Plotting Data for a Specific Week in NOVEMBER')
# Format the x-axis labels with abbreviated day names
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%a'))
axs[1].tick_params(axis='both', which='major', labelsize= 18)
# Adjust layout
plt.tight_layout()

# Display the plots
plt.show()


#######################ONE DAY####################

# # Filter the data for a specific day
# desired_date = pd.to_datetime('2013-07-20')  # pick the day to plot
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



# ############################## ONE HOUR##############
 
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




