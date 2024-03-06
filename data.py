import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import gc
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
                    if(len(distance_matrix)==66):
                        dfs.append(distance_matrix)
                        # Perform hierarchical clustering
                        #linkage_matrix = linkage(distance_matrix, method='average')

                        # Get the headers for labeling
                        #headers = df.columns

                        # Extract only the first word until a space from each header
                        #labels = [str(header).split()[0][5:] for header in headers]

                        # Plot the dendrogram with modified labels
                        #plt.figure(figsize=(10, 6))
                        #dendrogram(linkage_matrix, labels=labels, orientation='top', color_threshold=float('inf'))
                        #folder_name = os.path.basename(os.path.normpath(folder_path))
                        #plt.title(f'{folder_name} - {os.path.basename(file_path)[:4]}')
                        #plt.xlabel('Clusters')
                        #plt.ylabel('Distance')

                        # Show the plot
                        #plt.savefig(str(file_path).strip('.xlsx')+'.png')
                        #plt.close()
                        #print(f"Read {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Concatenate all dataframes into a single dataframe
    return dfs

def calculate_mantel_distance(dataframes):
    num_matrices = len(dataframes)
    distance_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(i, num_matrices):
            mantel_corr, _, _ = mantel.test(dataframes[i], dataframes[j])
            distance_matrix[i, j] = distance_matrix[j, i] = 1 - mantel_corr

    return pd.DataFrame(distance_matrix, index=range(num_matrices), columns=range(num_matrices))

# Specify the root folder path
#root_folder_path = r'D:\Documents\tesis\UPGMA2'

# Call the function
#result_dataframe = read_excel_files_in_folders(root_folder_path)
# Calculate Mantel distance matrix
#distance_matrix = calculate_mantel_distance(result_dataframe)

#print(distance_matrix)
#distance_matrix.to_excel(r'D:\Documents\tesis\prueba.xlsx')
# Perform hierarchical clustering
distance_matrix = pd.read_excel(r'D:\Documents\tesis\prueba.xlsx', index_col=0)
linkage_matrix = linkage(distance_matrix, method='average')

# Plot the dendrogram with modified labels
plt.figure(figsize=(10, 6))
label = [str(i) for i in range(1, len(linkage_matrix) + 2)]
dendrogram(linkage_matrix, labels=label, orientation='top', color_threshold=float('inf'))
plt.title(f'UPGMA')
plt.xlabel('Clusters')
plt.ylabel('Distance')

# Show the plot
plt.show()
plt.savefig(r'D:\Documents\tesis\prueba.svg')
#plt.savefig(str(file_path).strip('.xlsx')+'.png')
#gc.collect()

