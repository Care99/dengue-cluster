import math
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.spatial.distance import euclidean
#from tslearn.metrics import dtw
import os
import matplotlib as mplt; mplt.use('SVG',force=True)
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import MSTL
#Forecasting models
#Baseline Models 
from darts.models import NaiveMean,NaiveSeasonal,NaiveDrift,NaiveMovingAverage
#Statistical Models 
from darts.models import AutoARIMA,ExponentialSmoothing,AutoTheta,Prophet
#SKLearn-Like Models 
from darts.models import LinearRegressionModel,RandomForestModel
#PyTorch (Lightning)-based Models
from darts.models import RNNModel,NLinearModel,TCNModel
#Ensemble Models
from darts.models import NaiveEnsembleModel,RegressionEnsembleModel

from darts.metrics import accuracy,coefficient_of_variation,dtw_metric,mae,mape,precision,r2_score,rmse,smape
#from pmdarima.arima import auto_arima
script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
cluster_matriz_path = os.path.join(csv_path,'cluster_matriz')
matriz_ventana_path = os.path.join(csv_path,'matriz_ventana')
ts_historico_path = os.path.join(csv_path,'ts_historico')
departments = ['ALTO PARARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'Centro est','Centro norte','Centro sur','Chaco','CORDILLERA',
              'Metropolitano','PARAGUARI','Paraguay','PTE HAYES','SAN PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO PARAGUAY']
years = [2019,2020,2021,2022]
months = [
  "ENERO",
  "FEBRERO",
  "MARZO",
  "ABRIL",
  "MAYO",
  "JUNIO",
  "JULIO",
  "AGOSTO",
  "SEPTIEMBRE",
  "OCTUBRE",
  "NOVIEMBRE",
  "DICIEMBRE"
]
month_window = [
  "ENERO-FEBRERO-MARZO",
  "FEBRERO-MARZO-ABRIL",
  "MARZO-ABRIL-MAYO",
  "ABRIL-MAYO-JUNIO",
  "MAYO-JUNIO-JULIO",
  "JUNIO-JULIO-AGOSTO",
  "JULIO-AGOSTO-SEPTIEMBRE",
  "AGOSTO-SEPTIEMBRE-OCTUBRE",
  "SEPTIEMBRE-OCTUBRE-NOVIEMBRE",
  "OCTUBRE-NOVIEMBRE-DICIEMBRE",
  "NOVIEMBRE-DICIEMBRE-ENERO",
  "DICIEMBRE-ENERO-FEBRERO"
]
models = [
  'naive_mean',
  'naive_seasonal',
  'naive_drift',
  'naive_movingAverage',
  'auto_arima',
  'exponential_smoothing',
  'auto_theta',
  'prophet',
  'linear_regression_model',
  'random_forest_model',
  'rnn_model',
  'lstm_model',
  'nLinear_model',
  'tcn_model',
  'naive_ensemble_model',
  'regression_ensemble_model'
  ]
#departments = ['ALTO PARARANA']
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
def naive_mean(time_series,forecasted_values):
  data = time_series
  model = NaiveMean()
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def naive_seasonal(time_series,forecasted_values):
  data = time_series
  model = NaiveSeasonal(52)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def naive_drift(time_series,forecasted_values):
  data = time_series
  model = NaiveDrift()
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def naive_moving_average(time_series,forecasted_values):
  data = time_series
  model = NaiveMovingAverage(input_chunk_length=4)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def auto_arima(time_series,forecasted_values):
  data = time_series
  #train, test = model_selection.train_test_split(data)
  arima = AutoARIMA(data, error_action='ignore', trace=True, suppress_warnings=True,maxiter=10,seasonal=True,m=52,max_D=1,max_d=1,max_P=2,max_p=2,max_Q=2,max_q=2)
  generated_time_series = arima.predict(n_periods=forecasted_values)
  return generated_time_series.values()
def exponential_smoothing(time_series,forecasted_values):
  data = time_series
  model = ExponentialSmoothing(seasonal_periods=52)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def auto_theta(time_series,forecasted_values):
  data = time_series
  model = AutoTheta(season_length=52)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def prophet_forecast(time_series,forecasted_values):
  data = time_series
  model = Prophet(
    add_seasonalities=
    {
      'name': 'yearly_seasonality',  # (name of the seasonality component),
      'seasonal_periods': 52,  # (nr of steps composing a season),
      'fourier_order': 5,  # (number of Fourier components to use),
    }
  )
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def linear_regression(time_series,forecasted_values):
  data = time_series
  model = LinearRegressionModel(lags=52)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def random_forest(time_series,forecasted_values):
  data = time_series
  model = RandomForestModel(lags=52,n_estimators=100)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def rnn_forecast(time_series,forecasted_values):
  data = time_series
  model = RNNModel(
    model='RNN',
    hidden_size=25, 
    n_rnn_layers=2, 
    dropout=0.1, 
    batch_size=16, 
    n_epochs=100, 
    optimizer_kwargs={'lr':1e-3}, 
    random_state=42, 
    log_tensorboard=False, 
    force_reset=True
  )
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def lstm_forecast(time_series,forecasted_values):
  data = time_series
  model = RNNModel(
    model='LSTM',
    hidden_size=25, 
    n_rnn_layers=2, 
    dropout=0.1, 
    batch_size=16, 
    n_epochs=100, 
    optimizer_kwargs={'lr':1e-3}, 
    random_state=42, 
    log_tensorboard=False, 
    force_reset=True
  )
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def n_linear_forecast(time_series,forecasted_values):
  data = time_series
  model = NLinearModel(
    input_chunk_length=52,
    n_epochs=100,
    random_state=42,
    force_reset=True
  )
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def tcn_forecast(time_series,forecasted_values):
  data = time_series
  model = TCNModel(
    input_chunk_length=52,
    n_epochs=100,
    random_state=42,
    force_reset=True
  )
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def naive_ensemble_forecast(time_series,forecasted_values):
  data = time_series
  model = NaiveEnsembleModel(
    models=[
      NaiveMean(),
      NaiveSeasonal(K=52),
      NaiveDrift()
    ]
  )
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def regression_ensemble_forecast(time_series,forecasted_values):
  data = time_series
  model = RegressionEnsembleModel(
    base_models=[
      AutoARIMA(),
      ExponentialSmoothing(seasonal_periods=52),
      AutoTheta(season_length=52),
      Prophet(
        add_seasonalities=
        {
          'name': 'yearly_seasonality',  # (name of the seasonality component),
          'seasonal_periods': 52,  # (nr of steps composing a season),
          'fourier_order': 5,  # (number of Fourier components to use),
        }
      )
    ],
    regression_model=RandomForestModel(n_estimators=100)
  )
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
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

def load_time_series(path):
  df = pd.read_csv(path,header=None)
  time_series = df.to_numpy(dtype=float)
  return time_series

def get_historical_data(input_year,input_department):
  historical_time_series = []
  for year in range(initial_year,input_year):
    for month in months:
        filename = f'{input_department}.csv'
        path = os.path.join(ts_historico_path,{year},{month},filename)
        temp_ts = load_time_series(path)
    historical_time_series.append(temp_ts)
  return historical_time_series

def get_knn(input_year,input_month,input_department,neighbor_size):
  neighbors_ts = []
  neighbors = []
  # Find nearest neighbor for the given year
  filename = 'cluster_jerarquico.csv'
  path = os.path.join(cluster_matriz_path,month_window[input_month],filename)
  index = f'{input_department}-{input_year}-{input_year+1}'
  neighbors = find_nearest_neighbor(csv_path,index,neighbor_size)

  #Dado los años/departamentos mas cercanos, obtener sus ts
  for neighbor in neighbors:
    year = initial_year + int(neighbor/24)
    department = departments[int(neighbor)%24]
    filename = f'{departments[department]}.csv'
    path = os.path.join(ts_historico_path,f'{year}',f'{months[input_month]}',filename)
    temp_ts = load_time_series(path)
    neighbors_ts.append(temp_ts)
  neighbors_ts.reverse()
  knn_time_series = np.array(neighbors_ts,dtype=float).flatten()
  return knn_time_series

def get_cluster_clusters_knn(input_year,input_month,input_department,number_years,number_neighbors):
  neighbors_ts = []
  neighbors = []
  temp_ts = []
  # Find nearest neighbor for the given year
  filename = 'cluster_de_cluster.csv'
  path = os.path.join(cluster_matriz_path,month_window[input_month],filename)
  index = input_year - initial_year
  years = find_nearest_year(csv_path,index,number_years)

  #Dado los años/departamentos mas cercanos, obtener sus ts
  for year in years:
    filename = f'{year}-{year+1}.csv'
    path = os.path.join(cluster_matriz_path,month_window[input_month],filename)
    index = departments.index(input_department)
    neighbors = find_nearest_neighbor(path,index,number_neighbors)
    for neighbor in neighbors:
      department = departments[neighbor]
      filename = f'{departments[department]}.csv'
      path = os.path.join(ts_historico_path,f'{year}',f'{months[input_month]}',filename)
      temp_ts = load_time_series(path)
      neighbors_ts.append(temp_ts)
  neighbors_ts.reverse()
  knn_time_series = np.array(neighbors_ts,dtype=float).flatten()
  return knn_time_series

def generate_forecast(
    input_year,
    input_month,
    input_department,
    number_years,
    number_neighbors,
    values_to_forecast):
  #variables
  size_ts = len(original_time_series)
  neighbor_size=4
  original_time_series = []
  historical_time_series = get_historical_data(input_year,input_department)
  knn_time_series = get_knn(input_year,input_month,input_department,number_years*number_neighbors)
  cluster_clusters_knn = get_cluster_clusters_knn(input_year,input_month,input_department,number_years,number_neighbors)
  #Time series for projections
  sarima_historical_time_series = []
  knn_historical_time_series = []
  lstm_historical_time_series = []
  rnn_historical_time_series = []
  prophet_historical_time_series = []
  sarima_historical_time_series = []
  sarima_historical_time_series = []
  sarima_historical_time_series = []
  sarima_historical_time_series = []
  
  final_time_series = np.zeros(size_ts,dtype=float)
  final_time_series[:size_training_data] = original_time_series[:size_training_data]
  #final_time_series[size_training_data:] = forecast(knn_time_series,k,w,forecast_values)
  final_time_series[size_training_data:] = sarima_forecast(knn_time_series,size_training_data)
  final_time_series = add_zeros(final_time_series)
  
  #obtener error
  error_dist = logq(original_time_series, final_time_series)
  nueva_distancia = (error_dist,metric_name,final_time_series)
  return nueva_distancia

#variables
def project_time_series(k,n,month,forecasted_value):
  months_original_time_series=[
    ['OCTUBRE','2022'],
    ['NOVIEMBRE','2022'],
    ['DICIEMBRE','2022'],
    ['ENERO','2023'],
    ['FEBRERO','2023'],
    ['MARZO','2023'],
    ['ABRIL','2023'],
    ['MAYO','2023'],
    ['JUNIO','2023'],
    ['JULIO','2023'],
    ['AGOSTO','2023'],
    ['SEPTIEMBRE','2023']
  ]
  year = 2022
  best_error = np.inf
  best_k= np.inf
  best_w = np.inf
  #k=18
  #w=9
  #for k in range(1,30):
  #  for w in range(1,53):
  error_in_department = np.zeros(24)
  puntaje_funciones = np.zeros(len(conjunto_funciones))
  current_error = 0
  for input_department in departments:
    distancias = []
    threads = []
    original_time_series = []
    #Obtener el ts_original
    for month in months_original_time_series:
      input_year = int(month[1])
      input_month = months.index(month[0])
      filename = f'{input_department}.csv'
      path = os.path.join(ts_historico_path,f'{input_year}',f'{input_month}',filename)
      temp_ts = load_time_series(path)
      original_time_series.append(temp_ts)
    metric_index = 0
    for metric_name in conjunto_funciones:
      nueva_distancia = generate_forecast(input_year,input_department,metric_name,original_time_series,k,n,forecasted_value)
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
  #print(f'k:{k}\tw:{w}\terror:{current_error}')
  if(current_error < best_error):
    best_error = current_error
    best_k = k
    #best_w = w
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
  return [departments,error_in_department]