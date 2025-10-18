#step 1
import pandas as pd
import os
import matplotlib as mplt; mplt.use('SVG',force=True)
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram,linkage
import numpy as np
import math
from pandas import DataFrame
script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
ts_historico_path = os.path.join(csv_path,'ts_historico')

svg_path = os.path.join(script_directory,'svg')
months = ['ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']
years = [2019,2020,2021,2022,2023]
departments = ['ALTO_PARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'CENTRO_EST','CENTRO_NORTE','CENTRO_SUR','CHACO','CORDILLERA',
              'METROPOLITANO','PARAGUARI','PARAGUAY','PTE_HAYES','SAN_PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO_PARAGUAY']
def folder(year,month,department):
  # Read the Excel file
  excel_name = 'casos.csv'
  excel_file = os.path.join(csv_path,excel_name)
  data = pd.read_csv(excel_file)

  # Apply the filter
  # Create a folder with the specified format
  os.makedirs(ts_historico_path,exist_ok=True)
  year_path = os.path.join(ts_historico_path,str(year))
  os.makedirs(year_path,exist_ok=True)
  month_path = os.path.join(year_path,month)
  os.makedirs(month_path,exist_ok=True)

  last_day=[30,28,31,30,31,30,31,31,30,31,30,31]
  # Manage the time ranges
  # Iterate through filtered data and create subfolders
  # Initialize an empty DataFrame to store incidence data for the year
  incidence_data = pd.DataFrame()
  # Filter the data
  filtered_data = data[
     (data['disease'] == "DENGUE") 
     & (data['classification'] == "TOTAL") 
     & (data['name'] == department)]

  # Convert date column to pd.date
  filtered_data = filtered_data.copy()
  filtered_data['date'] = pd.to_datetime(filtered_data['date'], format='%Y-%m-%d')

  # Define the start and end dates
  start_date = pd.to_datetime(f"{year}-{months.index(month)+1}-1", format='%Y-%m-%d')
  if( months.index(month)+1 == 2 and year == 2020 ):
    end_day = last_day[months.index(month)] + 1
  else:
    end_day = last_day[months.index(month)]
  end_date = pd.to_datetime(f"{year}-{months.index(month)+1}-{end_day}", format='%Y-%m-%d')

  # Filter the data for the specified period
  range_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='left')]

  # Extract the incidence columns
  incidence_columns = [col for col in range_data.columns if 'incidence' in col]
  incidence_values = range_data[incidence_columns].values.flatten()  # Get incidence values as a flat array

  # Create a row with the department name and incidence values
  row_data = incidence_values.tolist()
  incidence_data = pd.DataFrame([row_data])

  # Save the DataFrame as a CSV file
  output_file_name = f'{department}.csv'

  output_file = os.path.join(month_path, output_file_name)
  incidence_data.to_csv(output_file, index=False)
  print(f"Saved: csv/{year}/{month}/{output_file}")

#10
def canberra(tseries1, tseries2):
  return distance.canberra(tseries1,tseries2)

#At first glance, Canberra metric given in the eqn 
#(10) [2,15] resembles Sørensen but normalizes the absolute 
#difference of the individual level. It is known to be very 
#sensitive to small changes near zero
#Bhattacharyya distance given in the eqn 
#(33), which is a value between 0 and 1, provides bounds on 
#the Bayes misclassification probability [23].
#26
def bhattacharyya(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
        value += math.sqrt(tseries1[i]*tseries2[i])
    value = - np.log(value)
    return value



# Function to load and process a single CSV file
def load_and_process_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path, header=None)
    # Extract the data starting from the second row and second column
    return df.iloc[0]

def create_cluster_clusters():
  # Initialize an empty list to store the DataFrames
  dfs = []
  # Sort files
  sorted_files=sorted(os.listdir(processed_data_path))
  # Process each uploaded file
  for filename in sorted_files:
      if filename.startswith('training_data'):
          # Load and process the CSV file
          print(filename)
          df = load_and_process_csv(os.path.join(processed_data_path, filename))
          # Append the DataFrame to the list
          dfs.append(df)

  # Concatenate all DataFrames along the rows
  combined_time_series = pd.concat(dfs, axis=0)

  # If you need the result as a NumPy array
  combined_ts_matrix = combined_time_series.values
  for i in range(len(combined_ts_matrix)):
    for j in range(len(combined_ts_matrix[0])):
        if(np.isnan(combined_ts_matrix[i][j])):
          combined_ts_matrix[i][j]=(combined_ts_matrix[i][j-1]+combined_ts_matrix[i][j+1])/2

  # Print the resulting matrix
  print(len(combined_ts_matrix[0]))
  print(len(combined_ts_matrix))

  os.makedirs(resultado_funciones_path,exist_ok=True)
  distance_matrix_size = 96
  resultado_funciones_total = np.zeros((distance_matrix_size,distance_matrix_size))
  for k in range(4):
    function_folder = os.path.join(resultado_funciones_path,bhattacharyya)
    os.makedirs(function_folder,exist_ok=True)
    distance_matrix_size = 24
    resultado_funciones = np.zeros((distance_matrix_size,distance_matrix_size))
    for i in range(int(len(combined_ts_matrix)/4)):
      for j in range(i+1,int(len(combined_ts_matrix)/4)):
        resultado_funciones[i,j] = bhattacharyya(combined_ts_matrix[(k*24)+i],combined_ts_matrix[(k*24)+j])
        if(np.isinf(resultado_funciones[i,j])):
            resultado_funciones[i,j] = 0
    for i in range(len(combined_ts_matrix)):
      for j in range(i+1,len(combined_ts_matrix)):
        resultado_funciones_total[i,j] = bhattacharyya(combined_ts_matrix[i],combined_ts_matrix[j])
        if(np.isinf(resultado_funciones_total[i,j])):
            resultado_funciones_total[i,j] = 0
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(resultado_funciones, method='average')
    # Get the headers for labeling
    headers = df.columns

    # Plot the dendrogram with modified labels
    plt.figure(figsize=(10, 10))
    labelsX = departments
    year = str(initial_year + k)
    print(f'bhattacharyya_{year}')
    dendrogram(linkage_matrix,labels=labelsX,orientation='top', color_threshold=0.7,leaf_rotation=90,leaf_font_size=7,)

    # Create sub folders
    csv_folder = os.path.join(function_folder,'csv')
    os.makedirs(csv_folder, exist_ok=True)
    svg_folder = os.path.join(function_folder,'svg')
    os.makedirs(svg_folder, exist_ok=True)

    # Plot graph
    plt.title(f'bhattacharyya-{year}')
    plt.xlabel('Departamento')
    plt.ylabel('Distancia')

    # Save the plot
    svg_file = f'bhattacharyya_{year}.svg'
    svg_folder = os.path.join(svg_folder,svg_file)
    plt.savefig(svg_folder)

    matriz_distancia = DataFrame(resultado_funciones)
    csv_file = f'bhattacharyya_{year}.csv'
    #csv_folder = os.path.join(csv_folder,csv_file)
    matriz_distancia.to_csv(os.path.join(csv_folder,csv_file))
    
    matriz_distancia = DataFrame(resultado_funciones_total + resultado_funciones_total.T - np.diag(np.diag(resultado_funciones_total)))
    csv_file = f'bhattacharyya_all.csv'
    matriz_distancia.to_csv(os.path.join(csv_folder,csv_file))
    
    #Close plot and finish
    plt.clf()
    plt.close()
    


  for folder in os.listdir(resultado_funciones_path):
    current_folder = os.path.join(resultado_funciones_path,folder,'csv')
    vector = []
    files = os.listdir(current_folder)
    for file in sorted(files):
      if not (file[-7:].startswith('all')):
        df = pd.read_csv(os.path.join(current_folder, file))
        vector.append(df.to_numpy().flatten())

    matriz_distancia = np.zeros((len(vector),len(vector)))
    for i in range(len(matriz_distancia)):
      for j in range(i+1,len(matriz_distancia)):
        dist = distance.euclidean(vector[i], vector[j])
        matriz_distancia[i,j] = dist
        matriz_distancia[j,i] = dist
    # Perform hierarchical clustering
    condensed_distance_matrix = distance.squareform(matriz_distancia)
    linkage_matrix = linkage(condensed_distance_matrix, method='average')


    # Get the headers for labeling
    headers = df.columns
    # Plot the dendrogram with modified labels
    plt.figure(figsize=(10, 10))
    labelsX = ["2020","2021","2022","2023"]
    year = str(initial_year + k)
    print(folder)
    dendrogram(linkage_matrix,labels=labelsX,orientation='top', color_threshold=0.7,leaf_rotation=90,leaf_font_size=7,)

    # Plot graph
    plt.title(f'{folder}')
    plt.xlabel('Year')
    plt.ylabel('Distance')

    # Save the plot
    svg_file = f'{folder}.svg'
    svg_folder = os.path.join(resultado_funciones_path,folder,svg_file)
    csv_file = f'{folder}.csv'
    csv_folder = os.path.join(resultado_funciones_path,folder,csv_file)
    plt.savefig(svg_folder)
    df_matriz_distancia = DataFrame(matriz_distancia)
    df_matriz_distancia.to_csv(csv_folder)
    #Close plot and finish
    plt.clf()
    plt.close()

for year in years:
   for month in months:
      for department in departments:
         folder(year,month,department)