# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:49:20 2023

@author: catar
"""
import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Clima Folder 
directory_path = r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Clustering de dados climaticos\clima_data"

#  all Excel files in the folder 
excel_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]

#  store dataframes for each cluster in each file
all_cluster_dataframes = []

# store mean dataframes for each file 
mean_dataframes = {}

# Loop through each Excel file 
for parameter_file in excel_files:
    parameter_file_path = os.path.join(directory_path, parameter_file)
    
    # Check if the parameter file is specified in the cluster count dictionary
    if parameter_file.endswith('.xlsx'):
        # Read the data from the Excel file, excluding the first two rows
        parameter_data = pd.read_excel(parameter_file_path, header=None, skiprows=2)
        
        # Extract the temperature data (columns 6 to the end)
        temp_cols = list(range(6, len(parameter_data.columns)))
        clima_data = parameter_data[temp_cols]
        
        # Normalize the clima_data using Z-score
        clima_data = (clima_data - clima_data.mean()) / clima_data.std()
        
        # Perform hierarchical clustering 
        num_clusters = 5   
        hierarchical_cluster = AgglomerativeClustering(n_clusters=num_clusters)
        cluster_labels = hierarchical_cluster.fit_predict(clima_data.T)  #  column-wise clustering
        
        # Create a DataFrame to store the cluster labels
        cluster_labels_df = pd.DataFrame({'Cluster_Labels': cluster_labels})
        
        # Store cluster labels for each column
        cluster_labels_df.index = clima_data.columns  # Set the index to column names
        cluster_labels_df = cluster_labels_df.T
        
        # Create a dictionary to store columns for each cluster
        cluster_columns = {}
        for cluster in range(num_clusters):
            cluster_columns[cluster] = clima_data.columns[cluster_labels == cluster]
        
        all_cluster_dataframes.append((parameter_file, cluster_labels_df, cluster_columns))

        # Create a mean DataFrame for this file
        mean_df = pd.DataFrame()
        
        # Extract year, month, day, and hour columns for the date
        year_col = 0  # Replace with the actual column index for year
        month_col = 1  # Replace with the actual column index for month
        day_col = 2  # Replace with the actual column index for day
        hour_col = 3  # Replace with the actual column index for hour
        
        # Create a timestamp column by combining year, month, day, and hour
        mean_df['Timestamp'] = parameter_data.apply(
            lambda row: f"{int(row[year_col])}-{int(row[month_col]):02d}-{int(row[day_col]):02d} {int(row[hour_col]):02d}:00:00",
            axis=1
        )

        for cluster, columns in cluster_columns.items():
            mean_column = clima_data[columns].mean(axis=1)
            mean_df[f'Cluster_{cluster}'] = mean_column

        mean_dataframes[parameter_file] = mean_df

# Create an Excel writer object to save separate sheets
output_excel_file = 'clustered_columns_with_mean_sheet.xlsx'  # Specify the desired output file name
with pd.ExcelWriter(output_excel_file) as writer:
    for parameter_file, cluster_labels_df, cluster_columns in all_cluster_dataframes:
        # Create unique sheet names for each file
        base_sheet_name = os.path.splitext(parameter_file)[0]
        
        # Save cluster labels as a separate sheet
        cluster_labels_df.to_excel(writer, sheet_name=f'Cluster_Labels_{base_sheet_name}', index=False)
        
        # Save cluster columns as separate sheets
        for cluster, columns in cluster_columns.items():
            cluster_data = clima_data[columns]
            cluster_data.insert(len(cluster_data.columns), 'Mean', cluster_data.mean(axis=1))  # Add a 'Mean' column
            cluster_data.to_excel(writer, sheet_name=f'Cluster_{cluster}_{base_sheet_name}', index=False)

# Create a separate Excel file for mean sheets
mean_output_excel_file = 'mean_sheets_with_date.xlsx'  # Specify the desired output file name for mean sheets
with pd.ExcelWriter(mean_output_excel_file) as mean_writer:
    for parameter_file, mean_df in mean_dataframes.items():
        base_sheet_name = os.path.splitext(parameter_file)[0]
        mean_df.to_excel(mean_writer, sheet_name=f'Mean_{base_sheet_name}', index=False)

print(f'Saved clustered columns with mean sheets to {output_excel_file}')
print(f'Saved mean sheets with date to {mean_output_excel_file}')







# import os
# import pandas as pd
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.preprocessing import MinMaxScaler

# # Clima Folder 
# directory_path = r"C:\Users\catar\OneDrive\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\TESE\Codigo atualizado - 16 de Novembro\Clustering de dados climaticos\clima_data"

# # All Excel files in the folder 
# excel_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]

# # Store dataframes for each cluster in each file
# all_cluster_dataframes = []

# # Store mean dataframes for each file 
# mean_dataframes = {}

# # Loop through each Excel file 
# for parameter_file in excel_files:
#     parameter_file_path = os.path.join(directory_path, parameter_file)
    
#     # Check if the parameter file is specified in the cluster count dictionary
#     if parameter_file.endswith('.xlsx'):
#         # Read the data from the Excel file, excluding the first two rows
#         parameter_data = pd.read_excel(parameter_file_path, header=None, skiprows=2)
        
#         # Extract the temperature data (columns 6 to the end)
#         temp_cols = list(range(6, len(parameter_data.columns)))
#         clima_data = parameter_data[temp_cols]
        
#         # Normalize the 'clima_data' using Min-Max scaling
#         scaler_min_max = MinMaxScaler()
#         clima_data_normalized = scaler_min_max.fit_transform(clima_data)
#         clima_data = pd.DataFrame(clima_data_normalized, columns=clima_data.columns)
        
#         # Perform hierarchical clustering 
#         num_clusters = 5   
#         hierarchical_cluster = AgglomerativeClustering(n_clusters=num_clusters)
#         cluster_labels = hierarchical_cluster.fit_predict(clima_data.T)  #  column-wise clustering
        
#         # Create a DataFrame to store the cluster labels
#         cluster_labels_df = pd.DataFrame({'Cluster_Labels': cluster_labels})
        
#         # Store cluster labels for each column
#         cluster_labels_df.index = clima_data.columns  # Set the index to column names
#         cluster_labels_df = cluster_labels_df.T
        
#         # Create a dictionary to store columns for each cluster
#         cluster_columns = {}
#         for cluster in range(num_clusters):
#             cluster_columns[cluster] = clima_data.columns[cluster_labels == cluster]
        
#         all_cluster_dataframes.append((parameter_file, cluster_labels_df, cluster_columns))

#         # Create a mean DataFrame for this file
#         mean_df = pd.DataFrame()
        
#         # Extract year, month, day, and hour columns for the date
#         year_col = 0  # Replace with the actual column index for year
#         month_col = 1  # Replace with the actual column index for month
#         day_col = 2  # Replace with the actual column index for day
#         hour_col = 3  # Replace with the actual column index for hour
        
#         # Create a timestamp column by combining year, month, day, and hour
#         mean_df['Timestamp'] = parameter_data.apply(
#             lambda row: f"{int(row[year_col])}-{int(row[month_col]):02d}-{int(row[day_col]):02d} {int(row[hour_col]):02d}:00:00",
#             axis=1
#         )

#         for cluster, columns in cluster_columns.items():
#             mean_column = clima_data[columns].mean(axis=1)
#             mean_df[f'Cluster_{cluster}'] = mean_column

#         mean_dataframes[parameter_file] = mean_df

# # Create an Excel writer object to save separate sheets
# output_excel_file = 'clustered_columns_with_mean_sheet.xlsx'  # Specify the desired output file name
# with pd.ExcelWriter(output_excel_file) as writer:
#     for parameter_file, cluster_labels_df, cluster_columns in all_cluster_dataframes:
#         # Create unique sheet names for each file
#         base_sheet_name = os.path.splitext(parameter_file)[0]
        
#         # Save cluster labels as a separate sheet
#         cluster_labels_df.to_excel(writer, sheet_name=f'Cluster_Labels_{base_sheet_name}', index=False)
        
#         # Save cluster columns as separate sheets
#         for cluster, columns in cluster_columns.items():
#             cluster_data = clima_data[columns]
#             cluster_data.insert(len(cluster_data.columns), 'Mean', cluster_data.mean(axis=1))  # Add a 'Mean' column
#             cluster_data.to_excel(writer, sheet_name=f'Cluster_{cluster}_{base_sheet_name}', index=False)

# # Create a separate Excel file for mean sheets
# mean_output_excel_file = 'mean_sheets_with_date.xlsx'  # Specify the desired output file name for mean sheets
# with pd.ExcelWriter(mean_output_excel_file) as mean_writer:
#     for parameter_file, mean_df in mean_dataframes.items():
#         base_sheet_name = os.path.splitext(parameter_file)[0]
#         mean_df.to_excel(mean_writer, sheet_name=f'Mean_{base_sheet_name}', index=False)

# print(f'Saved clustered columns with mean sheets to {output_excel_file}')
# print(f'Saved mean sheets with date to {mean_output_excel_file}')
