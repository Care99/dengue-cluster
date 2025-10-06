import math
import pandas as pd
from pmdarima import auto_arima
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.spatial.distance import euclidean
#from tslearn.metrics import dtw
import os
import matplotlib as mplt; mplt.use('SVG',force=True)
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import MSTL
#from pmdarima.arima import auto_arima
script_directory = os.getcwd()
processed_data_path = os.path.join(script_directory,'processed_data')
resultado_funciones_path = os.path.join(processed_data_path,'resultado_funciones')
departments = ['ALTO_PARARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'CENTRO_EST','CENTRO_NORTE','CENTRO_SUR','CHACO','CORDILLERA',
              'METROPOLITANO','PARAGUARI','PARAGUAY','PTE_HAYES','SAN_PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO_PARAGUAY']
#departments = ['ALTO_PARARANA']
conjunto_funciones = [ 
   "bhattacharyya",
]
initial_year = 2019
current_year = 2022
def logq(time_series,forecast):
  n = len(time_series)
  error = 0
  for i in range(n):
    numerator = forecast[i]
    denominator = time_series[i]
    if(numerator==0):
      numerator=0.001
    if(denominator==0):
      denominator=0.001
    value = numerator/denominator
    error = error + np.power(np.log(value),2)
  return error
def smape(time_series,forecast):
  numerator = 0.0
  denominator = 0.0
  error = 0.0
  n = len(time_series)
  for i in range(n):
    numerator = abs(time_series[i]-forecast[i])
    denominator = abs(time_series[i]) + abs(forecast[i])
    if(denominator==0):
      denominator = 0.0001
    error = error + (numerator/denominator)
  error = (error*100)/n
  return error

def remove_zeros(x):
  for i in range(int(len(x))):
    if(x[i]<=0):
      x[i]=0.01
  return x

def add_zeros(x):
  for i in range(int(len(x))):
    if(x[i]<=0.01):
      x[i]=0
  return x
# Function to detect and adjust outliers
def DAO(x):
    T = len(x)
    for i in range(3, T-3):
        mb = np.median([x[i-3], x[i-2], x[i-1]])
        ma = np.median([x[i+1], x[i+2], x[i+3]])
        if abs(x[i]) >= 4 * max(abs(mb), abs(ma)):
            x[i] = 0.5 * (x[i-1] + x[i+1])
    return x

# Function to stabilize variance using Box-Cox transformation
def stabilize_variance(x):
    x = remove_zeros(x)
    x, lam = boxcox(x)
    return x, lam


# Function to detrend a time series
def detrend(x):
  return np.diff(x)

def rank_order_centroid(k):
  weights = np.zeros(k)
  for j in range(k):
    inverted_sum = 0
    for i in range(j,k):
      inverted_sum = inverted_sum + (1/(i+1))
    weights[j] =  inverted_sum/k
  return weights

# Main function to perform CPTO-WNN time series forecasting
def forecast(x,k,p,n):
  T = len(x)
  distances = np.zeros(T-2*n-p)
  distances_m = np.zeros(T-2*n-p)
  s = np.zeros(n)
  forecast = np.zeros(n)
  b = np.zeros(n)
  weighted_value = 0
  indicator_function = 0
  # Step 1: Outliers adjustment
  x  = DAO(x)
  
  # Step 2: Variance stabilization
  g,lam = stabilize_variance(x[:T-n])
  
  # Step 3: Detrending
  h = detrend(g)
  #g_series = pd.Series(g, index=pd.date_range("1-1-2019", periods=len(g), freq="W"), name="Dengue")
  #h_decomp = seasonal.STL(g_series)
  #h_fit = h_decomp.fit()
  #h_trend = np.array(h_fit.trend.values,dtype=float)
  #h_seasonal = np.array(h_fit.seasonal.values,dtype=float)
  #h_remainder = np.array(h_fit.resid.values,dtype=float)
  #h = h_seasonal + h_remainder
  # Step 4: Distance vector
  for i in range(T-2*n-p):
    array1 = h[T-n-p-1:T-n]
    array2 = h[i:i+p]
    distances[i] = euclidean(array1,array2)

  # Step 5: Sorted distance
  distances_m = sorted(distances)

  #Setp 5: Neighborhood Set
  NS = distances.argsort()[:k]
  
  weights = rank_order_centroid(k)
  #Last known value before forecast
  Y = g[T-n-1]
  #Y = g[len(g)-1]
  for c in range(n):
    for j in range(k):
      weighted_value = 0
      detrending_value = 0
      indicator_function = 0
      for i in range(len(NS)):
        if(distances[NS[i]]==distances_m[j]):
          indicator_function = indicator_function + 1
          detrending_value = detrending_value + h[i+p+(c-1)]
      weighted_value = weights[j]/indicator_function
      s[c] = s[c] + weighted_value * detrending_value
    #Once we remove the trend by implementing
    # the Holt’s method, we forecast the seasonal and remainder components by
    # the k-NN regression method. 
    # The forecast of the trend component, the seasonal component, and the remainder
    # component are added to generate the final forecast
    #b[c] = s[c] + Y + h_trend[i+p+(c-1)]
    b[c] = s[c] + Y
    Y = b[c]
    forecast[c] = inv_boxcox(b[c],lam)
    if(np.isnan(forecast[c])):
      forecast[c] = 0
  return forecast

def cross_validate_knn(x,k_values,w_values,training_sets,n):
  size_ts = len(x)
  size_training_data = size_ts - n
  I = 1
  best_mape = np.inf
  best_forecast = np.zeros(size_ts)
  best_k = 0
  best_w = 0
  for k in range(1,k_values):
    for w in range(1,w_values):
      if( w + k <= size_ts - n * I - n + 1 ):
        generated_x = np.zeros(size_ts)
        generated_x[:size_training_data] 
        generated_x = forecast(training_sets,k_values,w_values,n)
        error_value = logq(x,generated_x)
        
        #Best case scenario of forecast is when error_value is 1
        #This happens when generated_x = x
        #logq = x/generated_x
        #logq = x/x
        #logq = 1
        if(abs(error_value-1)<abs(best_mape-1)):
          best_k = k
          best_w = w
          
          best_mape = error_value
          best_forecast = generated_x
          print(f'{input_year},{input_department},{metric_name},{k},{w},{error_value}')
        #if(error_value>=best_mape and error_value<(best_mape+best_mape/20)):
        #  print(f'{input_year},{input_department},{metric_name},{k},{w},{error_value}')
  #print(f'best_k={best_k},best_w={best_w}')
  return best_forecast,best_mape

def sarima_forecast(time_series,size_training_data):
  data = time_series
  #train, test = model_selection.train_test_split(data)
  arima = auto_arima(data, error_action='ignore', trace=True, suppress_warnings=True,maxiter=10,seasonal=True,m=52,max_D=1,max_d=1,max_P=2,max_p=2,max_Q=2,max_q=2)
  generated_time_series = arima.predict(n_periods=52)
  return generated_time_series[size_training_data:]
def plot_two_time_series(ts_original, ts_generado,department,year):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    size_ts = 52
    # Plot time series 1 and 2
    ax.plot(range(1, size_ts+1), ts_original, marker='o', linestyle='-', color='b', label='Time Series Original')
    ax.plot(range(1, size_ts+1), ts_generado, marker='s', linestyle='--', color='r', label='Time Series Generado')
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Time Series')
    ax.legend()
    # Show plot
    plt.tight_layout()
    #plt.show()
    plot_path = os.path.join(csv_data_path,f'{department}_{year}.svg')
    plt.savefig(plot_path)
    plt.clf()
    plt.close()

def plot_variance(ts,error_dist,department,year):
  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(10, 6))
  # Plot time series 1 and 2
  ax.plot(range(1, 53), ts, marker='o', linestyle='-', color='b', label='Time Series')
  # Add labels and title
  ax.set_xlabel('Week')
  ax.set_ylabel('Dissimilarity')
  ax.set_title(f'dissimilarity:{error_dist}')
  ax.legend()
  # Show plot
  plt.tight_layout()
  #plt.show()
  plot_path = os.path.join(csv_data_path,f'{department}_{year}_diss.svg')
  plt.savefig(plot_path)
  plt.clf()
  plt.close()

def find_nearest_neighbor(csv_path, index, num_neighbors):
  # Read the CSV file into a DataFrame
  df = pd.read_csv(csv_path, header=None, index_col=None, skiprows=1).iloc[:, 1:]
  # Extract the distance matrix
  matriz_distancia = df.values
  # Check the shape of the distance matrix
  if matriz_distancia.shape[0] != matriz_distancia.shape[1]:
      raise ValueError("The distance matrix is not square.")
  # Extract the distances for the given department index
  distances = matriz_distancia[index]
  # Get the indices of the nearest neighbors (excluding the department itself)
  nearest_indices = np.argsort(distances)
  return_indices = np.zeros(num_neighbors,dtype=int)
  i = 0
  for indice in nearest_indices:
    year = initial_year + int(indice/24)
    if(year<current_year):
      return_indices[i] = indice
      i = i + 1
      if( i == num_neighbors ):
        break
  return return_indices

def find_nearest_year(csv_path, index, num_neighbors):
  # Read the CSV file into a DataFrame
  df = pd.read_csv(csv_path, header=None, index_col=None, skiprows=1).iloc[:, 1:]
  # Extract the distance matrix
  matriz_distancia = df.values
  # Check the shape of the distance matrix
  if matriz_distancia.shape[0] != matriz_distancia.shape[1]:
      raise ValueError("The distance matrix is not square.")
  # Extract the distances for the given department index
  distances = matriz_distancia[index]
  # Get the indices of the nearest neighbors (excluding the department itself)
  nearest_indices = np.argsort(distances)
  return_indices = np.zeros(num_neighbors,dtype=int)
  i = 0
  for indice in nearest_indices:
      if(indice != index):
        return_indices[i] = initial_year + indice
        i = i + 1
        if( i == num_neighbors ):
          break
  return return_indices

#10700

def load_time_series(path,filename,index):
  df_path = os.path.join(path,filename)
  df = pd.read_csv(df_path)
  department_index = departments.index(index)
  time_series = np.array(df.to_numpy()[department_index:department_index+1,1:].flatten(),dtype=float)
  return time_series

def get_historical_data(original_time_series,training_size,input_year,input_department):
  neighbors_ts = []
  size_ts = len(original_time_series)
  historical_time_series = np.zeros(size_ts*(input_year-initial_year)+training_size,dtype=float)
  sliced_time_series = original_time_series[:training_size]
  for year in range(initial_year,input_year):
    filename = f'time_series_{year}.csv'
    temp_ts = load_time_series(csv_data_path,filename,input_department)
    historical_time_series[size_ts*(year-initial_year):size_ts*(year-initial_year+1)] = temp_ts
  historical_time_series[len(historical_time_series)-training_size:] = sliced_time_series
  return historical_time_series

def get_knn(original_time_series,input_year,input_department,metric_name,neighbor_size):
  neighbors_ts = []
  neighbors = []
  temp_ts = np.zeros(len(original_time_series))
  # Find nearest neighbor for the given year
  csv_path = os.path.join(cdc_matrix_diff_path,f'{metric_name}','csv',f'{metric_name}_all.csv')
  index = (24*(input_year-initial_year))+departments.index(input_department)
  neighbors = find_nearest_neighbor(csv_path,index,neighbor_size)

  #Dado los años/departamentos mas cercanos, obtener sus ts
  for neighbor in neighbors:
    year = initial_year + int(neighbor/24)
    department = departments[int(neighbor)%24]
    filename = f'time_series_{year}.csv'
    temp_ts = load_time_series(csv_data_path,filename,department)
    neighbors_ts.append(temp_ts)
  neighbors_ts.reverse()
  knn_time_series = np.array(neighbors_ts,dtype=float).flatten()
  return knn_time_series

def get_cluster_clusters_knn(original_time_series,input_year,input_department,metric_name,number_years,number_neighbors):
  neighbors_ts = []
  neighbors = []
  temp_ts = np.zeros(len(original_time_series))
  # Find nearest neighbor for the given year
  csv_path = os.path.join(cdc_matrix_diff_path,f'{metric_name}',f'{metric_name}.csv')
  index = input_year - initial_year
  years = find_nearest_year(csv_path,index,number_years)

  #Dado los años/departamentos mas cercanos, obtener sus ts
  for year in years:
    department_path = os.path.join(cdc_matrix_diff_path,f'{metric_name}','csv',f'{metric_name}_{year}.csv')
    index = departments.index(input_department)
    neighbors = find_nearest_neighbor(department_path,index,number_neighbors)
    for neighbor in neighbors:
      department = departments[neighbor]
      filename = f'time_series_{year}.csv'
      temp_ts = load_time_series(csv_data_path,filename,department)
      neighbors_ts.append(temp_ts)
  neighbors_ts.reverse()
  knn_time_series = np.array(neighbors_ts,dtype=float).flatten()
  return knn_time_series

def generate_forecast(input_year,input_department,metric_name,original_time_series,k,w):
  #variables
  size_ts = len(original_time_series)
  forecast_values = 40
  size_training_data = size_ts - forecast_values
  neighbor_size=4
  
  #historical_time_series = get_historical_data(original_time_series,size_training_data,input_year,input_department)
  #knn_time_series = get_knn(original_time_series,input_year,input_department,metric_name,neighbor_size)
  number_years=2
  number_neighbors=2
  knn_time_series = get_cluster_clusters_knn(original_time_series,input_year,input_department,metric_name,number_years,number_neighbors)
  final_time_series = np.zeros(size_ts,dtype=float)
  final_time_series[:size_training_data] = original_time_series[:size_training_data]
  final_time_series[size_training_data:] = forecast(knn_time_series,k,w,forecast_values)
  #final_time_series[size_training_data:] = sarima_forecast(knn_time_series,size_training_data)
  final_time_series = add_zeros(final_time_series)
  
  #obtener error
  error_dist = logq(original_time_series, final_time_series)
  nueva_distancia = (error_dist,metric_name,final_time_series)
  return nueva_distancia

#variables
years = [current_year]
best_error = np.inf
best_k= np.inf
best_w = np.inf
k=18
w=9
#for k in range(1,30):
#  for w in range(1,53):
error_in_department = np.zeros(24)
puntaje_funciones = np.zeros(len(conjunto_funciones))
current_error = 0
for input_year in years:
  for input_department in departments:
    distancias = []
    threads = []
    original_time_series = []
    #Obtener el ts_original
    filename = f'time_series_{input_year}.csv'
    original_time_series = load_time_series(csv_data_path,filename,input_department)
    metric_index = 0
    for metric_name in conjunto_funciones:
      nueva_distancia = generate_forecast(input_year,input_department,metric_name,original_time_series,k,w)
      distancias.append(nueva_distancia)
      print(input_department)
      print(nueva_distancia[2])
    #Resultados
    for i in range(len(conjunto_funciones)):
      puntaje_funciones[conjunto_funciones.index(distancias[i][1])] = puntaje_funciones[conjunto_funciones.index(distancias[i][1])] + distancias[i][0]
    sorted_distancias = sorted(distancias, key=lambda x: x[0])
    #for element in sorted_distancias:
    #    print(element[:2])
    generated_time_series = sorted_distancias[0][2]
    error_in_department[departments.index(input_department)] = error_in_department[departments.index(input_department)] + sorted_distancias[0][0]
    current_error = current_error + sorted_distancias[0][0]
    error = np.zeros(52)
    n = len(error)
    #plot_variance(error,sorted_distancias[0][0],input_department,input_year)
    plot_two_time_series(original_time_series, sorted_distancias[0][2],input_department,input_year)
  print(f'k:{k}\tw:{w}\terror:{current_error}')
  if(current_error < best_error):
    best_error = current_error
    best_k = k
    best_w = w
    #print(f'best_k={best_k},best_w={best_w},error={best_error}')
        


print('==========================================')
print('Puntaje final')
print(f'best_k={best_k},best_w={best_w},error={best_error}')
i = 0
for metric in conjunto_funciones:
   print(f'{metric}={puntaje_funciones[i]}')
   i = i + 1

print('==========================================')
print('Ganador')
print(min(puntaje_funciones))
print('==========================================')
print('Error por ciudad:')
for i in range(24): 
  print(f'{departments[i]}:{error_in_department[i]}')  