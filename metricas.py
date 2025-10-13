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
csv_raw_data_path = os.path.join(csv_path,'raw_data')
csv_incidence_ts_path = os.path.join(csv_raw_data_path,'incidence_ts')
csv_historical_ts_path = os.path.join(csv_raw_data_path,'incidence_ts')
csv_windowed_ts_path = os.path.join(csv_raw_data_path,'windowed_ts')

svg_path = os.path.join(script_directory,'svg')
svg_raw_data_path = os.path.join(csv_path,'raw_data')
svg_incidence_ts_path = os.path.join(csv_raw_data_path,'incidence_ts')
svg_historical_ts_path = os.path.join(csv_raw_data_path,'incidence_ts')
svg_windowed_ts_path = os.path.join(csv_raw_data_path,'windowed_ts')
def folder(start_month,end_month,filename,path):

    # Read the Excel file
    excel_name = 'casos.csv'
    excel_file = os.path.join(csv_path,excel_name)
    data = pd.read_csv(excel_file)

    # Apply the filter

    # Remove repeated values
    name_data = data.drop_duplicates(subset='name')

    # Create a folder with the specified format
    os.makedirs(path,exist_ok=True)

    last_day=[30,28,31,30,31,30,31,31,30,31,30,31]
    # Manage the time ranges
    # Iterate through filtered data and create subfolders
    for year in range(2019, 2023):
      rows = []
    # Initialize an empty DataFrame to store incidence data for the year
      incidence_data = pd.DataFrame()
      for index, row in name_data.iterrows():
        # Filter the data
        department_name = row['name']
        filtered_data = data[(data['disease'] == "DENGUE") &
                              (data['classification'] == "TOTAL") &
                              (data['name'] == department_name)]

        # Convert date column to pd.date
        filtered_data = filtered_data.copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], format='%Y-%m-%d')

        # Define the start and end dates
        fixed_year = year
        start_date = pd.to_datetime(f"{year}-{start_month}-1", format='%Y-%m-%d')
        if( end_month < start_month ): 
          fixed_year = fixed_year + 1
        if( end_month == 2 and year == 2020 ):
          end_day = last_day[end_month-1] + 1
        end_day = last_day[end_month-1]
        end_date = pd.to_datetime(f"{fixed_year}-{end_month}-{end_day}", format='%Y-%m-%d')

        # Filter the data for the specified period
        range_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='left')]

        # Extract the incidence columns
        incidence_columns = [col for col in range_data.columns if 'incidence' in col]
        incidence_values = range_data[incidence_columns].values.flatten()  # Get incidence values as a flat array

        # Create a row with the department name and incidence values
        row_data = incidence_values.tolist()
        columns = [f'week_{i+1}' for i in range(len(incidence_values))]
        incidence_data = pd.DataFrame(row_data, columns=columns)

        # Save the DataFrame as a CSV file
        output_file_name = f'{filename}_{department_name}_{year}.csv'
        output_file_path = os.path.join(path, output_file_name)
        incidence_data.to_csv(output_file_path, index=False)
        print(f"Saved: {output_file_path}")
months = ['ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']
initial_year = 2019
start_month = 9
end_month = 8
filename = 'time_series'
path = os.path.join(csv_path,'raw_data_incidence')
folder(start_month,end_month,filename,path)

start_month = 1
end_month = 12
filename = 'time_series'
path = os.path.join(csv_path,'raw_data_historical')
folder(start_month,end_month,filename)

for i in range(1,13):
  month1 = i
  month2 = i+1
  month3 = i=2
  fmonth1 = month1
  fmonth2 = month2
  fmonth3 = month3
  index_month1 = month1
  index_month2 = month2
  index_month3 = month3
  if(fmonth1<10): fmonth1 = f'0{fmonth1}'
  if(fmonth2<10): fmonth2 = f'0{fmonth2}'
  if(fmonth3<10): fmonth3 = f'0{fmonth3}'
  if(fmonth1>12): fmonth1 = fmonth1%12
  if(fmonth2>12): fmonth2 = fmonth2%12
  if(fmonth3>12): fmonth3 = fmonth3%12
  if(index_month1>12): index_month1 = index_month1%12
  if(index_month2>12): index_month2 = index_month2%12
  if(index_month3>12): index_month3 = index_month3%12
  index_month1 = index_month1 - 1
  index_month2 = index_month2 - 1
  index_month3 = index_month3 - 1
  filename = 'time_series'
  path = os.path.join(csv_path,f'raw_data_{fmonth1}_{fmonth2}_{fmonth3}_{months[index_month1]}_{months[index_month2]}_{months[index_month3]}')
  folder(start_month,end_month,filename,path)

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
    df = pd.read_csv(file_path, header=0, index_col=0)
    # Extract the data starting from the second row and second column
    return df.iloc[:, 0:]

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
  labelsX = ['ALTO PARARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'Centro est','Centro norte','Centro sur','Chaco','CORDILLERA',
              'Metropolitano','PARAGUARI','Paraguay','PTE HAYES','SAN PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO PARAGUAY']
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