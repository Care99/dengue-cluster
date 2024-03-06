import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Read Excel file into a DataFrame
file_path = r'D:\Documents\tesis\UPGMA\SAN PEDRO\2019_1-3.xlsx'
df = pd.read_excel(file_path, index_col=0)  # Assuming the first column contains the dates
print(df)
distance_matrix = df.values

# Replace NaN values with 0
distance_matrix = df.fillna(0).values

# Perform hierarchical clustering
linkage_matrix = linkage(distance_matrix, method='average')

# Get the headers for labeling
headers = df.columns

# Extract only the first word until a space from each header
labels = [str(header).split()[0] for header in headers]

# Plot the dendrogram with modified labels
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=labels, orientation='top', color_threshold=float('inf'))
plt.title('UPGMA Dendrogram')
plt.xlabel('Clusters')
plt.ylabel('Distance')

# Show the plot
plt.show()