import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import mantel
def read_excel_files_in_folders(root_folder):
    # List to store dataframes from each Excel file
    dfs = []

    # Walk through the root folder and its subfolders
    for folder_path, _, file_names in os.walk(root_folder):
        for file_name in file_names:
            # Check if the file is an Excel file
            if file_name.endswith('.xlsx'):
                file_path = os.path.join(folder_path, file_name)            
                # Read the Excel file into a dataframe
                try:
                    df = pd.read_excel(file_path, index_col=0)
                    distance_matrix = df.values
                    # Replace NaN values with 0
                    distance_matrix = df.fillna(0).values
                    distance_matrix = squareform(distance_matrix,checks=False)
                    dfs.append(distance_matrix)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Concatenate all dataframes into a single dataframe
    return dfs

def calculate_mantel_distance(dataframes):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_name= 'cluster_matrix.xlsx'
    data_directory= os.path.join(script_directory,data_name)
    num_matrices = len(dataframes)
    distance_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(i, num_matrices):
            mantel_corr, _, _ = mantel.test(dataframes[i], dataframes[j])
            distance_matrix[i, j] = distance_matrix[j, i] = 1 - mantel_corr
    mantel_data_frame = pd.DataFrame(distance_matrix, index=range(num_matrices), columns=range(num_matrices))
    mantel_data_frame.to_excel(data_directory,engine='xlsxwriter')
    return mantel_data_frame

def data():
    # Specify the root folder path
    script_directory = os.path.dirname(os.path.abspath(__file__))
    root_folder_name= 'clusters'
    root_folder_directory= os.path.join(script_directory,root_folder_name)

    # Call the function
    result_dataframe = read_excel_files_in_folders(root_folder_directory)
    # Calculate Mantel distance matrix
    distance_matrix = calculate_mantel_distance(result_dataframe)
    print(distance_matrix)
    linkage_matrix = linkage(distance_matrix, method='average')
    print(linkage_matrix)
    # Plot the dendrogram with modified labels
    plt.figure(figsize=(10, 6))
    label = [str(i) for i in range(1, len(linkage_matrix) + 2)]
    dendrogram(linkage_matrix, labels=label, orientation='top', color_threshold=float('inf'))
    plt.title(f'UPGMA')
    plt.xlabel('Clusters')
    plt.ylabel('Distance')

    # Show the plot
    plt.show()
    fig_name='cluster_cluster.svg'
    fig_directory=os.path.join(script_directory,fig_name)
    plt.savefig(fig_directory)
    plt.clf()

