import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import datetime
def graph():
    index_time_series=0
    # Walk through the root folder and its subfolders
    script_directory = os.path.dirname(os.path.abspath(__file__))
    root_folder_name= 'raw_clusters'
    root_folder= os.path.join(script_directory,root_folder_name)
    cluster_folder_name= 'clusters'
    cluster_folder= os.path.join(script_directory,cluster_folder_name)
    os.makedirs(cluster_folder)
    list_of_departments=[os.listdir(os.path.join(root_folder,os.listdir(root_folder)[0]))]
    labels = list_of_departments
    for folder_path in os.listdir(root_folder):
        weeks_in_year=datetime.date(int(folder_path), 12, 31).isocalendar()[1]
        print(folder_path)
        print(weeks_in_year)
        list_time_series = []
        for file_name in os.listdir(os.path.join(root_folder,folder_path)):
            # Check if the file is an Excel file
            if file_name.endswith('.xlsx'):
                file_path = os.path.join(folder_path, file_name)
            # Read the Excel file into a dataframe
            try:
                excel_path= os.path.join(root_folder,file_path)
                df = pd.read_excel(excel_path, index_col=0)
                # Replace NaN values with 0
                np.nan_to_num(df,copy=True,nan=0,posinf=None,neginf=None)
                time_series = df['incidence'].values
                print(len(time_series))
                list_time_series.append(time_series)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Step 2: Choose a distance metric and compute the distance matrix
        list_time_series = np.array(list_time_series)
        distance_metric = 'euclidean'
        distances = pdist(list_time_series, metric=distance_metric)
        # Step 4: Create a new DataFrame for the upper triangle with extra columns and rows for labels

        # Create a DataFrame with NaN values
        result_df = pd.DataFrame(index=labels, columns=labels, dtype=float)

        # Fill the upper triangle with distance values
        result_df.values[np.triu_indices_from(result_df, k=1)] = distances
        result_df.fillna(0, inplace=True)
        # Display or save the result DataFrame
        print(result_df)

        #Store the range data in the subfolder
        output_file_path = os.path.join(cluster_folder, f"{folder_path}.xlsx")
        result_df.to_excel(output_file_path, index=False, engine='xlsxwriter')
        print(f"Fecha terminado: {output_file_path}")
        output_file_path= output_file_path.strip('.xlsx')

        # Perform hierarchical clustering
        linkage_matrix = linkage(result_df, method='average')
        print(linkage_matrix)
        # Get the headers for labeling
        headers = df.columns

        # Extract only the first word until a space from each header
        #labels = [str(header).split()[0][5:] for header in headers]

        # Plot the dendrogram with modified labels
        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix, orientation='top', color_threshold=float('inf'))

        # Plot graph
        folder_name = os.path.basename(os.path.normpath(folder_path))
        plt.title(f'{folder_name} - {file_name[:4]}')
        plt.xlabel('Departamento')
        plt.ylabel('Distance')

        # Save the plot
        plt.savefig(f'{output_file_path}.svg')
        
        #Close plot and finish
        plt.clf()
        print(f"Image saved {output_file_path}.svg")