import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
def graph():
    # Walk through the root folder and its subfolders
    script_directory = os.path.dirname(os.path.abspath(__file__))
    root_folder_name= 'raw_clusters'
    root_folder= os.path.join(script_directory,root_folder_name)
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

                    # Perform hierarchical clustering
                    linkage_matrix = linkage(distance_matrix, method='average')

                    # Get the headers for labeling
                    headers = df.columns

                    # Extract only the first word until a space from each header
                    labels = [str(header).split()[0][5:] for header in headers]

                    # Plot the dendrogram with modified labels
                    plt.figure(figsize=(10, 6))
                    dendrogram(linkage_matrix, labels=labels, orientation='top', color_threshold=float('inf'))

                    # Plot graph
                    folder_name = os.path.basename(os.path.normpath(folder_path))
                    plt.title(f'{folder_name} - {file_name[:4]}')
                    plt.xlabel('Series de Tiempo')
                    plt.ylabel('Distance')

                    # Save the plot
                    plt.savefig(str(file_path).strip('.xlsx')+'.png')
                    
                    #Close plot and finish
                    plt.close()
                    print(f"Image saved {folder_name} - {file_name.strip('.xlsx')}")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")