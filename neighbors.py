from generar_cluster import get_k_n_n as c_get_knn
from generar_cluster_jerarquico import get_k_n_n as cj_get_knn
from generar_cluster_de_cluster import get_k_n_n as cdc_get_knn
import math
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.spatial.distance import euclidean
#from tslearn.metrics import dtw
import os
from time import process_time
import matplotlib as mplt
import classifiers
from classifiers import evaluate_models, fill_na; mplt.use('SVG',force=True)
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import MSTL
from darts import concatenate
#Forecasting models
#Baseline Models 
from darts.models import NaiveMean,NaiveSeasonal,NaiveDrift,NaiveMovingAverage
#Statistical Models 
from darts.models import AutoARIMA,ExponentialSmoothing,AutoTheta,Prophet
#SKLearn-Like Models 
from darts.models import LinearRegressionModel,RandomForestModel
#PyTorch (Lightning)-based Models
from darts.models import RNNModel,NLinearModel,TCNModel,NBEATSModel
#Ensemble Models
from darts.models import NaiveEnsembleModel,RegressionEnsembleModel

from darts.metrics import accuracy,coefficient_of_variation,dtw_metric,mae,mape,precision,r2_score,rmse,smape
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import torch
torch.set_float32_matmul_precision('medium')
#from pmdarima.arima import auto_arima
script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
cluster_matriz_path = os.path.join(csv_path,'cluster_matriz')
matriz_ventana_path = os.path.join(csv_path,'matriz_ventana')
ts_historico_path = os.path.join(csv_path,'ts_historico')
departments = [
  'ALTO_PARANA',
  'AMAMBAY',
  'ASUNCION'
  'CAAGUAZU',
  'CENTRAL',
  'CENTRO_EST',
  'CENTRO_NORTE',
  'CENTRO_SUR',
  'CHACO',
  'CORDILLERA',
  'METROPOLITANO',
  'PARAGUARI',
  'PARAGUAY',
  'PTE_HAYES',
  'SAN_PEDRO',
  'CANINDEYU',
  'CONCEPCION',
  'ITAPUA',
  'MISIONES',
  'BOQUERON',
  'GUAIRA',
  'CAAZAPA',
  'NEEMBUCU',
  'ALTO_PARAGUAY'
  ]
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
#departments = ['ALTO PARARANA']
conjunto_funciones = [
   "bhattacharyya",
]
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
initial_year = 2019
current_year = 2022
def get_real_data(input_department):
  original_time_series = []
  for month_year in months_original_time_series:
    filename = f'{input_department}.csv'
    path = os.path.join(ts_historico_path,str(month_year[1]),month_year[0],filename)
    temp_ts = load_time_series(path)
    original_time_series.extend(temp_ts)
  return original_time_series
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
def naive_drift(time_series,forecasted_values):
  data = time_series
  model = NaiveDrift()
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values().flatten()

def auto_arima(time_series,forecasted_values):
  data = time_series
  #train, test = model_selection.train_test_split(data)
  model = AutoARIMA(season_length=52)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values().flatten()
def linear_regression(time_series,forecasted_values):
  data = time_series
  model = LinearRegressionModel(lags=52)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values().flatten()
def lstm_forecast(time_series,forecasted_values):
  data = time_series
  model = RNNModel(
    model='LSTM',
    input_chunk_length=1,
    hidden_dim=25, 
    n_rnn_layers=2, 
    dropout=0.0, 
    batch_size=16,
    n_epochs=100, 
    optimizer_kwargs={'lr':1e-3}, 
    random_state=42, 
    log_tensorboard=False, 
    force_reset=True,
    pl_trainer_kwargs={
      "accelerator": "gpu",
      "devices": [0]
    },
  )
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values().flatten()
models = [
  naive_drift,
  auto_arima,
  linear_regression,
  lstm_forecast,
  ]
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
  time_series = df.to_numpy(dtype=float).flatten().tolist()
  return time_series

def get_historical_data(input_department):
  historical_time_series = []
  for year in range(initial_year,2023):
    if(year==2019):
      for month in ["OCTUBRE","NOVIEMBRE","DICIEMBRE"]:
        filename = f'{input_department}.csv'
        path = os.path.join(ts_historico_path,str(year),month,filename)
        temp_ts = load_time_series(path)
        for value in temp_ts:
          historical_time_series.append(value)
    if(year==2022):
      for month in ["ENERO","FEBRERO","MARZO","ABRIL","MAYO","JUNIO","JULIO","AGOSTO","SEPTIEMBRE"]:
        filename = f'{input_department}.csv'
        path = os.path.join(ts_historico_path,str(year),month,filename)
        temp_ts = load_time_series(path)
        for value in temp_ts:
          historical_time_series.append(value)
    else:
      for month in months:
        filename = f'{input_department}.csv'
        path = os.path.join(ts_historico_path,str(year),month,filename)
        temp_ts = load_time_series(path)
        for value in temp_ts:
          historical_time_series.append(value)
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
    input_department,
    number_years,
    number_neighbors,
    months_to_forecast,
    classification,
    model):
  #variables
  historical_time_series = get_historical_data(input_department)
  projected_time_series = []
  #Time series for projections
  months = [['JULIO',2022],
            ['AGOSTO',2022],
            ['SEPTIEMBRE',2022],
            ['OCTUBRE',2022],
            ['NOVIEMBRE',2022],
            ['DICIEMBRE',2022],
            ['ENERO',2023],
            ['FEBRERO',2023],
            ['MARZO',2023],
            ['ABRIL',2023],
            ['MAYO',2023],
            ['JUNIO',2023],
            ['JULIO',2023],
            ['AGOSTO',2023],
            ['SEPTIEMBRE',2023],
            ['OCTUBRE',2023]]
  for i in range(3,len(months)-1,months_to_forecast):
    size_ts = 0
    next_time_series = []
    for j in range(i-3,i):
      path_next = os.path.join(ts_historico_path,f'{months[i+1][1]}',f'{months[i+1][0]}',f'{input_department}.csv')
      temp_ts = load_time_series(path_next)
      size_ts += len(temp_ts)
      next_time_series.extend(next_time_series)
    historical_time_series = historical_time_series[size_ts:]
    time = 0
    start_time = process_time()
    match classification.__qualname__:
      case 'historical':
        time_series = TimeSeries.from_values(historical_time_series)
        forecasted_values = model(time_series,size_ts)
      case 'c_get_knn' | 'cj_get_knn' | 'cdc_get_knn':
        time_series = concatenate(classification(months[i][0],input_department,number_years,number_neighbors),axis=1).mean(axis=1)
        forecasted_values = model(time_series,size_ts)
      case 'CART' | 'RANDOM_FOREST' | 'TAN':
        forecasted_values = classification(historical_time_series,size_ts)
    end_time = process_time()
    time += end_time - start_time
    projected_time_series.extend(forecasted_values)
    historical_time_series.extend(next_time_series)
    historical_time_series = historical_time_series[size_ts:]
  save_time_series_as_csv(input_department,projected_time_series,model.__qualname__,classification.__qualname__,months_to_forecast)
  #plot_scatter(historical_time_series,projected_time_series,input_department,model,classification.__qualname__,months_to_forecast)
  #plot_histogram(historical_time_series,projected_time_series,input_department,model,classification.__qualname__,months_to_forecast)
  save_error(input_department,projected_time_series,model,classification.__qualname__,months_to_forecast)
  save_time(input_department,time,model,classification.__qualname__,months_to_forecast)
def save_time(department,time,model,classification,months_to_forecast):
  output_file_name = f'{department}_execution_time.txt'
  path = os.path.join(csv_path,'forecast',classification,model,f'{months_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file = os.path.join(path, output_file_name)
  with open(output_file, 'w') as f:
    f.write(f'{time}')
  print(f"Saved: {output_file}. Elapsed time: {time}")
def save_time_series_as_csv(department,time_series,model,classification,months_to_forecast):
  incidence_data = pd.DataFrame(time_series)
  # Save the DataFrame as a CSV file
  output_file_name = f'{department}.csv'
  path = os.path.join(csv_path,'forecast',classification,model,f'{months_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file = os.path.join(path, output_file_name)
  incidence_data.to_csv(output_file, header=False, index=False)
  print(f"Saved: {output_file}")
def plot_scatter(actual,predicted,input_department,model,classification,months_to_forecast):
  plt.figure(figsize=(10, 6))
  plt.scatter(actual,predicted)
  plt.title(f'Scatter plot for {input_department} using {model.__qualname__} - {classification}')
  plt.xlabel('Actual values')
  plt.ylabel('Predicted values')
  plt.legend()
  path = os.path.join(csv_path,'forecast',classification,model.__qualname__,f'{months_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file_name = f'{input_department}.svg'
  output_file = os.path.join(path, output_file_name)
  plt.savefig(output_file)
  plt.close()
  print(f"Saved: {output_file}")
def plot_histogram(actual,predicted,input_department,model,classification,months_to_forecast):
  plt.figure(figsize=(10, 6))
  errors = rmse(actual,predicted)
  plt.hist(actual,predicted)
  plt.title(f'Histogram for {input_department} using {model.__qualname__} - {classification}')
  plt.xlabel('Error')
  plt.ylabel('Frequency')
  plt.legend()
  path = os.path.join(csv_path,'forecast',classification,model.__qualname__,f'{months_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file_name = f'{input_department}.svg'
  output_file = os.path.join(path, output_file_name)
  plt.savefig(output_file)
  plt.close()
  print(f"Saved: {output_file}")
def save_error(input_department,time_series,model,classification,months_to_forecast):
  original_time_series = []
  for month_year in months_original_time_series:
    filename = f'{input_department}.csv'
    path = os.path.join(ts_historico_path,str(month_year[1]),month_year[0],filename)
    temp_ts = load_time_series(path)
    original_time_series.extend(temp_ts)
  path = os.path.join(csv_path,'forecast',classification,model.__qualname__,f'{months_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  mae_error = mae(original_time_series,time_series)
  output_file_name = f'{input_department}_mae.txt'
  output_file = os.path.join(path, output_file_name)
  write_error(output_file,mae_error)
  mape_error = mape(original_time_series,time_series)
  output_file_name = f'{input_department}_mape.txt'
  output_file = os.path.join(path, output_file_name)
  write_error(output_file,mape_error)
  rmse_error = rmse(original_time_series,time_series)
  output_file_name = f'{input_department}_rmse.txt'
  output_file = os.path.join(path, output_file_name)
  write_error(output_file,rmse_error)
  smape_error = smape(original_time_series,time_series)
  output_file_name = f'{input_department}_smape.txt'
  output_file = os.path.join(path, output_file_name)
  write_error(output_file,smape_error)
def write_error(output_file, error):
  with open(output_file, 'w') as f:
    f.write(f'{error}')
  print(f"Saved: {output_file}")
#variables
def project_time_series(k,n,forecasted_value):
  for input_department in departments:
    input_year=2022
    generate_forecast(input_year,input_department,k,n,forecasted_value)
number_of_neighbors=4
number_of_years=2
months_to_forecast=1
project_time_series(number_of_neighbors,number_of_years,months_to_forecast)
months_to_forecast=2
project_time_series(number_of_neighbors,number_of_years,months_to_forecast)
months_to_forecast=4
project_time_series(number_of_neighbors,number_of_years,months_to_forecast)