import pandas as pd
import os
import numpy as np 

def folder():
    # Script directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Read the Excel file
    excel_name = 'casos.csv'
    excel_file = os.path.join(script_directory,excel_name)
    data = pd.read_csv(excel_file)

    # Apply the filter
    departments_data = data[data['level'] == "Department"]

    # Remove repeated values
    departments_data = departments_data.drop_duplicates(subset='name')

    # Create a folder with the specified format
    folder_name = os.path.join(script_directory,'raw_clusters')
    os.makedirs(folder_name)

    # Iterate through filtered data and create subfolders
    month_range = (1, 12)
    for year in range(2020,2025):
        subfolder_name = f'{year}' 
        subfolder_path = os.path.join(folder_name, subfolder_name)
        os.makedirs(subfolder_path)
        
        for index, row in departments_data.iterrows():
            # Filter the data
            department_name = row['name']
            filtered_data = data[(data['disease'] == "DENGUE") & 
                            (data['classification'] == "TOTAL") &
                            (data['name'] == department_name) &
                            (data['level'] == "Department")]
            
            # Convert date column to pd.date
            filtered_data.loc[:, 'date'] = pd.to_datetime(filtered_data['date'], format='%Y-%m-%d')

            # Manage the time ranges
            start_month, end_month = month_range
            start_date = pd.to_datetime(f"{year}-{start_month}-1", format='%Y-%m-%d')
            end_date = pd.to_datetime(f"{year}-{end_month}-1", format='%Y-%m-%d') + pd.offsets.MonthEnd(0)
            # Get the ranged data
            range_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='left')]    
            
            #Store the range data in the subfolder
            output_file_name = f'{department_name}.xlsx'
            output_file_path = os.path.join(subfolder_path, output_file_name)
            range_data.to_excel(output_file_path, index=False, engine='xlsxwriter')
            print(f"{output_file_path}")