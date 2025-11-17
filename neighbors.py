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
from darts.models import RNNModel,NLinearModel,TCNModel,NBEATSModel
#Ensemble Models
from darts.models import NaiveEnsembleModel,RegressionEnsembleModel

from darts.metrics import accuracy,coefficient_of_variation,dtw_metric,mae,mape,precision,r2_score,rmse,smape
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
#from pmdarima.arima import auto_arima
script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
cluster_matriz_path = os.path.join(csv_path,'cluster_matriz')
matriz_ventana_path = os.path.join(csv_path,'matriz_ventana')
ts_historico_path = os.path.join(csv_path,'ts_historico')
departments = ['ALTO_PARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'CENTRO_EST','CENTRO_NORTE','CENTRO_SUR','CHACO','CORDILLERA',
              'METROPOLITANO','PARAGUARI','PARAGUAY','PTE_HAYES','SAN_PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO_PARAGUAY']
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
def naive_mean(time_series,forecasted_values):
  data = time_series
  model = NaiveMean()
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series.values()
def naive_seasonal(time_series,forecasted_values):
  data = time_series
  model = NaiveSeasonal(K=52)
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
  model = NaiveMovingAverage(input_chunk_length=12)
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
      AutoARIMA(error_action='ignore', trace=True, suppress_warnings=True,maxiter=10,seasonal=True,m=52,max_D=1,max_d=1,max_P=2,max_p=2,max_Q=2,max_q=2),
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
models = [
  naive_mean,
  naive_seasonal,
  naive_drift,
  #naive_moving_average,
  auto_arima,
  exponential_smoothing,
  auto_theta,
  prophet_forecast,
  linear_regression,
  random_forest,
  rnn_forecast,
  lstm_forecast,
  n_linear_forecast,
  tcn_forecast,
  naive_ensemble_forecast,
  regression_ensemble_forecast
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
  projected_historical_time_series = []
  projected_knn_time_series = []
  projected_cluster_clusters_knn = []
  for model in models:
    for month in months:
      current_time_series = []
      path = os.path.join(ts_historico_path,f'{input_year}',f'{month}',f'{input_department}.csv')
      current_time_series = load_time_series(path)
      size_ts = len(current_time_series)
      historical_time_series = historical_time_series[size_ts:]+[current_time_series]
      knn_time_series = knn_time_series[size_ts:]+[current_time_series]
      cluster_clusters_knn = cluster_clusters_knn[size_ts:]+[current_time_series]
      projected_historical_time_series.append(model.__func__(historical_time_series,values_to_forecast))
      projected_knn_time_series.append(model.__func__(knn_time_series,values_to_forecast))
      projected_cluster_clusters_knn.append(model.__func__(knn_time_series,values_to_forecast))
    save_time_series_as_csv(input_department,projected_historical_time_series,model,'HISTORICAL')
    save_time_series_as_csv(input_department,projected_knn_time_series,model,'KNN')
    save_time_series_as_csv(input_department,projected_cluster_clusters_knn,model,'CLUSTER_CLUSTERS')
    original_time_series = TimeSeries.from_values(original_time_series)
    projected_historical_time_series = TimeSeries.from_values(projected_historical_time_series)
    projected_knn_time_series = TimeSeries.from_values(projected_knn_time_series)
    projected_cluster_clusters_knn = TimeSeries.from_values(projected_cluster_clusters_knn)
    save_time_series_as_svg(input_department,projected_historical_time_series,model,'HISTORICAL')
    save_time_series_as_svg(input_department,projected_knn_time_series,model,'KNN')
    save_time_series_as_svg(input_department,projected_cluster_clusters_knn,model,'CLUSTER_CLUSTERS')
    save_error(input_department,projected_historical_time_series,model,'HISTORICAL')
    save_error(input_department,projected_knn_time_series,model,'KNN')
    save_error(input_department,projected_cluster_clusters_knn,model,'CLUSTER_CLUSTERS')
def save_time_series_as_csv(department,time_series,model,classification):
  incidence_data = pd.DataFrame(time_series)
  # Save the DataFrame as a CSV file
  output_file_name = f'{department}.csv'
  path = os.path.join(csv_path,'forecast',{classification},{model})
  os.makedirs(path,exist_ok=True)
  output_file = os.path.join(path, output_file_name)
  incidence_data.to_csv(output_file, header=False, index=False)
  print(f"Saved: csv/forecast/{classification}/{model}/{output_file_name}")
#variables
def project_time_series(k,n,forecasted_value):
  for input_department in departments:
    for month in months_original_time_series:
      input_year = int(month[1])
      input_month = months.index(month[0])
      generate_forecast(input_year,input_month,input_department,k,n,forecasted_value)