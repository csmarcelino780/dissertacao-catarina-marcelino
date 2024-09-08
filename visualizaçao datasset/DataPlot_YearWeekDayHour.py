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
day_array=data["DateTime"].values 
net_load_M80_array=data["MT_080"].values
plt.figure(figsize=(20,8)) # plot size 
plt.plot(day_array , net_load_M80_array)


################ONE WEEK#####################

# Define  week
start_date = pd.to_datetime('2012-02-01')
end_date = start_date + pd.DateOffset(weeks=1) # end date is one week after the start date 

# Filter the data for the week , only contain the values f MT_080 for the dates chosen 
filtered_data = data[(data['DateTime'] >= start_date) & (data['DateTime'] < end_date)]

# Plot the filtered data
plt.figure(figsize=(16,8))
plt.plot(filtered_data['DateTime'], filtered_data['MT_080'])

# Add labels and a title
plt.xlabel('Datetime')
plt.ylabel('load')
plt.title('Plotting Data for a Specific Week')

# Display the plot
plt.show()



#######################ONE DAY####################

# Filter the data for a specific day
desired_date = pd.to_datetime('2012-05-01')  # pick the day to plot
filtered_data = data[data['DateTime'].dt.date == desired_date.date()] #filter the data from the day picked

# Plot the filtered data
plt.figure(figsize=(16,8)) # plot size 
plt.plot(filtered_data['DateTime'],filtered_data['MT_080'] )

# Add labels and a title
plt.xlabel('Datetime')
plt.ylabel('load')
plt.title('Plotting Data for a Specific Day')

# Display the plot
plt.show()



############################## ONE HOUR##############
 
desired_hour = pd.to_datetime('2012-01-15 23:30:00')   #pick hour to plot 
filtered_data = data[data['DateTime'].dt.floor('H') == desired_hour.floor('H')] # filter the data form the hour chosen 

# Plot the filtered data
plt.figure(figsize=(16,8))
plt.plot(filtered_data['DateTime'], filtered_data['MT_080'])

# Add labels and a title
plt.xlabel('Datetime')
plt.ylabel('load')
plt.title('Plotting Data for a Specific Hour')

# Display the plot
plt.show()




