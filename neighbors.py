import math
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.spatial.distance import euclidean
#from tslearn.metrics import dtw
import os
import matplotlib as mplt
import classifiers
from classifiers import evaluate_models, fill_na; mplt.use('SVG',force=True)
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
  #'ALTO_PARANA',
  #'AMAMBAY',
  'ASUNCION'
  #'CAAGUAZU',
  #'CENTRAL',
  #'CENTRO_EST',
  #'CENTRO_NORTE',
  #'CENTRO_SUR',
  #'CHACO',
  #'CORDILLERA',
  #'METROPOLITANO',
  #'PARAGUARI',
  #'PARAGUAY',
  #'PTE_HAYES',
  #'SAN_PEDRO',
  #'CANINDEYU',
  #'CONCEPCION',
  #'ITAPUA',
  #'MISIONES',
  #'BOQUERON',
  #'GUAIRA',
  #'CAAZAPA',
  #'NEEMBUCU',
  #'ALTO_PARAGUAY'
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
  return generated_time_series.values()

def auto_arima(time_series,forecasted_values):
  data = time_series
  #train, test = model_selection.train_test_split(data)
  model = AutoARIMA(season_length=52)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def linear_regression(time_series,forecasted_values):
  data = time_series
  model = LinearRegressionModel(lags=52)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
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
  return generated_time_series.values()
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
    input_year,
    input_department,
    number_years,
    number_neighbors,
    months_to_forecast):
  #variables
  classifications = [
        ['HISTORICAL',[]],
        ['KNN',[]],
        ['CLUSTER_CLUSTERS',[]],
        ['CART',[]],
        ['RANDOM_FOREST',[]],
        ['KNN_CLASSIFIER',[]],
        ['TAN',[]]
      ]
  projected_classifications = [
        ['HISTORICAL',[]],
        ['KNN',[]],
        ['CLUSTER_CLUSTERS',[]],
        ['CART',[]],
        ['RANDOM_FOREST',[]],
        ['KNN_CLASSIFIER',[]],
        ['TAN',[]]
      ]
  classifications[0][1] = get_historical_data(input_department)
  classifications[3][1],classifications[4][1],classifications[5][1],classifications[6][1] = evaluate_models(classifications[0][1],number_years,number_neighbors)
  #Time series for projections
  months = [['OCTUBRE',2022],
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
  for model in models:
    for i in range(len(months)-1):
      classifications[1][1] = get_knn(input_year,i,input_department,number_years*number_neighbors)
      classifications[2][1] = get_cluster_clusters_knn(input_year,i,input_department,number_years,number_neighbors)
      path = os.path.join(ts_historico_path,f'{months[i][1]}',f'{months[i][0]}',f'{input_department}.csv')
      current_time_series = load_time_series(path)
      size_ts = 0
      for j in range(months_to_forecast):
        path_next = os.path.join(ts_historico_path,f'{months[i+1][1]}',f'{months[i+1][0]}',f'{input_department}.csv')
        next_time_series = load_time_series(path_next)
        size_ts += len(next_time_series)
      for j in range(len(classifications)):
        classifications[j][1].extend(current_time_series)
        classifications[j][1] = classifications[j][1][size_ts:]
        forecasted_values = model(TimeSeries.from_values(classifications[j][1]),size_ts)
        projected_classifications[j][1].extend(forecasted_values)
    for i in range(len(projected_classifications)):
      save_time_series_as_csv(input_department,projected_classifications[i][1],model.__qualname__,projected_classifications[i][0])
      save_time_series_as_svg(input_department,projected_classifications[i][1],model,projected_classifications[i][0])
      save_error(input_department,projected_classifications[i][1],model,projected_classifications[i][0])
def save_time_series_as_csv(department,time_series,model,classification):
  incidence_data = pd.DataFrame(time_series)
  # Save the DataFrame as a CSV file
  output_file_name = f'{department}.csv'
  path = os.path.join(csv_path,'forecast',classification,model)
  os.makedirs(path,exist_ok=True)
  output_file = os.path.join(path, output_file_name)
  incidence_data.to_csv(output_file, header=False, index=False)
  print(f"Saved: csv/forecast/{classification}/{model}/{output_file_name}")
def save_time_series_as_svg(input_department,time_series,model,classification):
  plt.figure(figsize=(10, 6))
  plt.plot(time_series, label='Forecasted Incidence', color='blue')
  plt.title(f'Forecasted Incidence for {input_department} using {model.__qualname__} - {classification}')
  plt.xlabel('Time')
  plt.ylabel('Incidence')
  plt.legend()
  path = os.path.join(csv_path,'forecast',classification,model.__qualname__)
  os.makedirs(path,exist_ok=True)
  output_file_name = f'{input_department}.svg'
  output_file = os.path.join(path, output_file_name)
  plt.savefig(output_file)
  plt.close()
  print(f"Saved: csv/forecast/{classification}/{model.__qualname__}/{output_file_name}")
def save_error(input_department,time_series,model,classification):
  original_time_series = []
  for month_year in months_original_time_series:
    filename = f'{input_department}.csv'
    path = os.path.join(ts_historico_path,str(month_year[1]),month_year[0],filename)
    temp_ts = load_time_series(path)
    for value in temp_ts:
      original_time_series.append(value)
  logq_error = logq(original_time_series,time_series)
  path = os.path.join(csv_path,'forecast',classification,model.__qualname__)
  os.makedirs(path,exist_ok=True)
  output_file_name = f'{input_department}_error.txt'
  output_file = os.path.join(path, output_file_name)
  with open(output_file, 'w') as f:
    f.write(f'LogQ Error: {logq_error}\n')
  print(f"Saved: csv/forecast/{classification}/{model.__qualname__}/{output_file_name}")
#variables
def project_time_series(k,n,forecasted_value):
  for input_department in departments:
    input_year=2022
    generate_forecast(input_year,input_department,k,n,forecasted_value)

project_time_series(4,2,1)
