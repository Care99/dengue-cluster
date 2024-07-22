import math
import concurrent.futures
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error as mape
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.spatial.distance import euclidean
from statsmodels.tsa.stl._stl import STL
import threading
from tslearn.metrics import dtw
import os
import matplotlib as mplt; mplt.use('SVG',force=True)
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import MSTL
#from pmdarima.arima import auto_arima
script_directory = os.getcwd()
processed_data_path = os.path.join(script_directory,'processed_data')
resultado_funciones_path = os.path.join(processed_data_path,'resultado_funciones')
departments = ['ALTO PARARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'Centro est','Centro norte','Centro sur','Chaco','CORDILLERA',
              'Metropolitano','PARAGUARI','Paraguay','PTE HAYES','SAN PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO PARAGUAY']
conjunto_funciones = [ 
   "bhattacharyya",
   "canberra",
]
def remove_zeros(x):
  for i in range(len(x)):
    if(x[i]<=0):
      x[i]=0.001
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
  distances = np.zeros(T-p)
  distances_m = np.zeros(T-p)
  s = np.zeros(n)
  forecast = np.zeros(n)
  b = np.zeros(n)
  weighted_value = 0
  indicator_function = 0
  # Step 1: Outliers adjustment
  #x  = DAO(x)
  
  # Step 2: Variance stabilization
  g,lam = stabilize_variance(x)
  
  # Step 3: Detrending
  h = detrend(g)
  #g_series = pd.Series(g, index=pd.date_range("1-1-1959", periods=len(g), freq="W"), name="Dengue")
  #h_decomp = STL(g_series)
  #h = h_decomp.fit()
  # Step 4: Distance vector
  for i in range(T-p):
    array1 = h[T-p-1:T-1]
    array2 = h[i:i+p]
    distances[i] = dtw(array1,array2)

  # Step 5: Sorted distance
  distances_m = sorted(distances)

  #Setp 5: Neighborhood Set
  NS = distances.argsort()[:k]
  
  weights = rank_order_centroid(k)
  Y = g[T-1]
  for c in range(n):
    for j in range(k):
      weighted_value = 0
      detrending_value = 0
      indicator_function = 0
      for i in range(len(NS)):
        if(distances[NS[i]]==distances_m[j]):
          indicator_function = indicator_function + 1
          detrending_value = detrending_value + h[(i+p+(c-1))%(T-1)]
      weighted_value = weights[j]/indicator_function
      s[c] = s[c] + weighted_value * detrending_value
    b[c] = s[c] + Y
    Y = b[c]
    forecast[c] = inv_boxcox(b[c],lam)
    if(np.isnan(forecast[c])):
      if(c>2):
        forecast[c] = (forecast[c-1]+forecast[c-2])/2
      else:
        forecast[c] = 0
  return forecast

def cross_validate_knn(x,k_values,w_values,training_sets,n):
  best_mape = np.inf
  best_forecast = np.zeros(n)
  sorted_forecast = []
  best_k = 0
  best_w = 0
  for k in range(1,k_values):
    for w in range(1,w_values):
      for training_set in training_sets:
        if((w+k<=len(x)) and (len(x)-w>0)):
          generated_x = forecast(training_set,k,w,n)
          error_value = mean_squared_error(x,generated_x)
          #print(f'{input_year},{input_department},{metric_name},{k},{w},{error_value}')
          if(error_value<best_mape):
            best_k = k
            best_w = w
            best_forecast = generated_x
            best_mape = error_value
            print(f'{input_year},{input_department},{metric_name},{k},{w},{error_value}')
            sorted_forecast.append(best_forecast)
          if(error_value>best_mape and error_value<(best_mape+best_mape/10)):
            print(f'{input_year},{input_department},{metric_name},{k},{w},{error_value}')
  if(len(sorted_forecast)>=5):
    for i in range(n):
      best_forecast[i] = (sorted_forecast[-1][i]+sorted_forecast[-2][i]+sorted_forecast[-3][i]+sorted_forecast[-4][i]+sorted_forecast[-5][i])/5
    if(mean_squared_error(x,best_forecast) < best_mape):
      best_mape = mean_squared_error(x,best_forecast)
    else:
      best_forecast = np.array(sorted_forecast.pop(),dtype=float)
  print(f'best_k={best_k},best_w={best_w}')
  return best_forecast,best_mape
def plot_two_time_series(ts_original, ts_generado,department,year):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot time series 1 and 2
    ax.plot(range(1, 53), ts_original, marker='o', linestyle='-', color='b', label='Time Series Original')
    ax.plot(range(1, 53), ts_generado, marker='s', linestyle='--', color='r', label='Time Series Generado')
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Time Series')
    ax.legend()
    # Show plot
    plt.tight_layout()
    #plt.show()
    plot_path = os.path.join(processed_data_path,f'{department}_{year}.svg')
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
  plot_path = os.path.join(processed_data_path,f'{department}_{year}_diss.svg')
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
    nearest_indices = np.argsort(distances)[0:num_neighbors + 1]
    return nearest_indices
#10700

def generate_forecast(input_year,input_department,metric_name,original_time_series):
  #variables
  neighbors_ts = []
  neighbors = []
  neighbor_size=5

  # Find nearest neighbor for the given year
  csv_path = os.path.join(resultado_funciones_path,f'{metric_name}','csv',f'{metric_name}_all.csv')
  index = (24*(input_year-2019))+departments.index(input_department)
  neighbors = find_nearest_neighbor(csv_path,index,neighbor_size)

  #Dado los años/departamentos mas cercanos, obtener sus ts
  for neighbor in neighbors:
    year = 2019 + int(neighbor/24)
    department = departments[(neighbor)%24]
    if(year<2022):
      df_path = os.path.join(processed_data_path,f'time_series_{year}.csv')
      df = pd.read_csv(df_path)
      i = departments.index(department)
      timeseries = df.to_numpy()[i:i+1,1:].flatten()
      neighbors_ts.append(timeseries)
  knn_time_series = np.array(neighbors_ts,dtype=float)

  #From testing it was found that the best value of
  #k tends to be 1 or 2, but values of k onwards gets 
  #identical or close error values to the best error value
  #Best value of w tends to be (len(x)-k)
  maximum_k = 50
  maximum_w = len(original_time_series)
  forecast_values = 52
  #print(f'input_year:{input_year},input_department:{input_department},metric:{metric_name}')
  final_time_series,error_dist = cross_validate_knn(original_time_series,maximum_k,maximum_w,knn_time_series,forecast_values)
  #final_time_series = forecast(knn_time_series[0],maximum_k,maximum_w,forecast_values)
  #print(f'input_year:{input_year},input_department:{input_department},metric:{metric_name},error_dist:{error_dist}')
  #obtener la distancia MSE
  #error_dist = mean_squared_error(original_time_series, final_time_series)
  nueva_distancia = (error_dist,metric_name,final_time_series)
  #print(distancias[-1])
  #plot_two_time_series(original_time_series, final_time_series)
  return nueva_distancia
#variables


years = [2022]
error_in_department = np.zeros(24)
puntaje_funciones = np.zeros(len(conjunto_funciones))

for input_year in years:
   for input_department in departments:
        distancias = []
        threads = []
        #Obtener el ts_original
        df_path = os.path.join(processed_data_path,f'time_series_{input_year}.csv')
        df = pd.read_csv(df_path)
        department_index = departments.index(input_department)
        original_time_series = np.array(df.to_numpy()[department_index:department_index+1,1:].flatten(),dtype=float)
        metric_index = 0
        for metric_name in conjunto_funciones:
          nueva_distancia = generate_forecast(input_year,input_department,metric_name,original_time_series)
          distancias.append(nueva_distancia)
        #Resultados
        for i in range(len(conjunto_funciones)):
          puntaje_funciones[conjunto_funciones.index(distancias[i][1])] = puntaje_funciones[conjunto_funciones.index(distancias[i][1])] + distancias[i][0]
        sorted_distancias = sorted(distancias, key=lambda x: x[0])
        #for element in sorted_distancias:
        #    print(element[:2])
        generated_time_series = sorted_distancias[0][2]
        error_in_department[departments.index(input_department)] = error_in_department[departments.index(input_department)] + sorted_distancias[0][0]
        error = np.zeros(52)
        for week in range(len(original_time_series)):
          error[week] = math.pow((original_time_series[week]-generated_time_series[week]),2)
        plot_variance(error,sorted_distancias[0][0],input_department,input_year)
        plot_two_time_series(original_time_series, sorted_distancias[0][2],input_department,input_year)
        print(f'input_year:{input_year},input_department:{input_department},metric:{sorted_distancias[0][1]},error:{sorted_distancias[0][0]}')
        #print(sorted_distancias[0])

print('==========================================')
print('Puntaje final')
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
