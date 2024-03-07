import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt


# Read data from Excel file
excel_file = r'C:\Users\Spooky\Desktop\tesis\datasource\datos.xlsx' 
data = pd.read_excel(excel_file)

# filtros
date_filter = '2019-01-05'
disease_filter = 'DENGUE'
classification_filter = 'TOTAL'
level_filter = 'Department'

# Filtrado
filtered_data = data[(data['date'] == date_filter) & 
                     (data['disease'] == disease_filter) & 
                     (data['classification'] == classification_filter) &
                     (data['level'] == level_filter)]


# Extract the column for UPGMA clustering
column_name = 'incidence'
distances = filtered_data[[column_name]]

#labels
labels = filtered_data['name']
#filtered_data.reset_index(drop=True, inplace=True)

# Calculate the absolute differences directly
distances = distances.abs().fillna(0)

#matriz
pairwise_distances = pdist(distances, metric='euclidean')
distance_matrix = squareform(pairwise_distances)
print("Complete Distance Matrix:")
print(distance_matrix)

# Perform UPGMA clustering
linked = linkage(distance_matrix, method='average')
print(filtered_data)
# Plot the dendrogram
dendrogram(linked, labels=filtered_data['name'].tolist(), orientation='top', distance_sort='descending')
plt.title(f'UPGMA Dendrogram - Date: {date_filter}')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()