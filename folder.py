import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform
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
    month_ranges = [(1, 12)]

    for index, row in departments_data.iterrows():
        subfolder_name = row['name'] 
        subfolder_path = os.path.join(folder_name, subfolder_name)
        os.makedirs(subfolder_path)
        # Filter the data
        filtered_data = data[(data['disease'] == "DENGUE") & 
                        (data['classification'] == "TOTAL") &
                        (data['name'] == subfolder_name) &
                        (data['level'] == "Department")]
        
        # Convert date column to pd.date
        filtered_data.loc[:, 'date'] = pd.to_datetime(filtered_data['date'], format='%Y-%m-%d')
        
        for year in range(2019, 2025):
            for month_range in month_ranges:
                # Manage the time ranges
                start_month, end_month = month_range
                start_date = pd.to_datetime(f"{year}-{start_month}-1", format='%Y-%m-%d')
                end_date = pd.to_datetime(f"{year}-{end_month}-1", format='%Y-%m-%d') + pd.offsets.MonthEnd(0)
                # Get the ranged data
                range_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='left')]    
                if not range_data.empty:
                    # Step 2: Choose a distance metric and compute the distance matrix
                    distance_metric = 'euclidean'
                    distances = pdist(range_data['incidence'].values.reshape(-1, 1), metric=distance_metric)

                    # Step 3: Convert the condensed distance matrix to a square distance matrix
                    distance_matrix = squareform(distances)

                    # Step 4: Create a new DataFrame for the upper triangle with extra columns and rows for labels
                    labels = range_data['date']

                    # Create a DataFrame with NaN values
                    result_df = pd.DataFrame(index=labels, columns=labels, dtype=float)

                    # Fill the upper triangle with distance values
                    result_df.values[np.triu_indices_from(result_df, k=1)] = distances

                    # Display or save the result DataFrame
                    print(result_df)

                    #Store the range data in the subfolder
                    output_file_path = os.path.join(subfolder_path, f'{year}_{start_month}-{end_month}.xlsx')
                    result_df.to_excel(output_file_path, index=False, engine='xlsxwriter')
                    print(f"Fecha terminado: {year}/{month_range}")

        print(f"Departamento terminado: {subfolder_name} \n")


