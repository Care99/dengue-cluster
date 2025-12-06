from generar_cluster import get_k_n_n as c_get_knn
from generar_cluster_jerarquico import get_k_n_n as cj_get_knn
from generar_cluster_de_cluster import get_k_n_n as cdc_get_knn
import math
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
#from tslearn.metrics import dtw
import os
from time import process_time
import matplotlib as mplt
from classifiers import CART, RANDOM_FOREST, TAN
mplt.use('SVG',force=True)
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import MSTL
from darts import concatenate
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers import InvertibleMapper
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
  'ASUNCION',
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
# Apply log transformation (ensure all values > 0)
def safe_log(x):
    return np.log1p(x)  # log(1 + x) handles zeros

def safe_exp(x):
    return np.expm1(x)  # exp(x) - 1 inverse

# Transform the data
log_transformer = InvertibleMapper(
    fn= safe_log,
    inverse_fn= safe_exp
)
def naive_drift(time_series,forecasted_values):
  data = time_series
  model = NaiveDrift()
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series

def auto_arima(time_series,forecasted_values):
  data = time_series
  #train, test = model_selection.train_test_split(data)
  model = AutoARIMA(season_length=4)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series
def linear_regression(time_series,forecasted_values):
  data = time_series
  model = LinearRegressionModel(lags=4)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series
def lstm_forecast(time_series,forecasted_values):
  data = time_series
  model = RNNModel(
    model='LSTM',
    input_chunk_length=12,
    output_chunk_length=forecasted_values,
    hidden_dim=25, 
    n_rnn_layers=1, 
    dropout=0.0, 
    batch_size=2,
    n_epochs=50, 
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
  return generated_time_series
models = [
  naive_drift,
  auto_arima,
  linear_regression,
  lstm_forecast,
  ]

def load_time_series(path):
  df = pd.read_csv(path,header=None)
  time_series = df.to_numpy(dtype=float).flatten().tolist()
  return time_series

def get_historical_data(input_department):
  historical_time_series = []
  for year in range(initial_year,2023):
    match year:
      case 2019:
        for month in ["OCTUBRE","NOVIEMBRE","DICIEMBRE"]:
          filename = f'{input_department}.csv'
          path = os.path.join(ts_historico_path,str(year),month,filename)
          temp_ts = load_time_series(path)
          for value in temp_ts:
            historical_time_series.append(value)
      case 2022:
        for month in ["ENERO","FEBRERO","MARZO","ABRIL","MAYO","JUNIO","JULIO","AGOSTO","SEPTIEMBRE"]:
          filename = f'{input_department}.csv'
          path = os.path.join(ts_historico_path,str(year),month,filename)
          temp_ts = load_time_series(path)
          for value in temp_ts:
            historical_time_series.append(value)
      case _:
        for month in months:
          filename = f'{input_department}.csv'
          path = os.path.join(ts_historico_path,str(year),month,filename)
          temp_ts = load_time_series(path)
          for value in temp_ts:
            historical_time_series.append(value)
  return historical_time_series

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
            ['SEPTIEMBRE',2023]]
  for i in range(0,len(months),months_to_forecast):
    size_ts = 0
    next_time_series = []
    for j in range(i,i+months_to_forecast):
      path_next = os.path.join(ts_historico_path,f'{months[j][1]}',f'{months[j][0]}',f'{input_department}.csv')
      temp_ts = load_time_series(path_next)
      size_ts += len(temp_ts)
      next_time_series.extend(temp_ts)
    time = 0
    start_time = process_time()
    match classification.__qualname__:
      case 'get_historical_data':
        time_series = TimeSeries.from_values(historical_time_series)
        scaled_time_series = log_transformer.transform(time_series)
        scaled_forecast = model(time_series,size_ts)
        forecasted_values = log_transformer.inverse_transform(scaled_forecast).values().flatten() 
      case 'get_k_n_n':
        temp_ts = historical_time_series[:-12]
        nearest_neighbors = classification(months[i][0],input_department,number_years,number_neighbors)
        forecasted_values = []
        for neighbor in nearest_neighbors:
          time_series = []
          time_series.extend(temp_ts)
          time_series.extend(neighbor.values().flatten())
          scaled_time_series = log_transformer.transform(TimeSeries.from_values(time_series))
          scaled_forecast = model(scaled_time_series,size_ts)
          forecast = log_transformer.inverse_transform(scaled_forecast)
          forecasted_values.append(forecast)
        forecasted_values = concatenate(forecasted_values,axis=1).mean(axis=1)
        forecasted_values = forecasted_values.values().flatten()
      case 'CART' | 'RANDOM_FOREST' | 'TAN':
        time_series = TimeSeries.from_values(historical_time_series)
        scaled_time_series = log_transformer.transform(time_series)
        scaled_forecast = classification(scaled_time_series,size_ts)
        forecasted_values = log_transformer.inverse_transform(scaled_forecast)
        forecasted_values = forecasted_values.values().flatten()
    projected_time_series.extend(forecasted_values)
    historical_time_series.extend(next_time_series)
    historical_time_series = historical_time_series[size_ts:]
  historical_time_series = TimeSeries.from_values(historical_time_series[:-len(projected_time_series)])
  projected_time_series = TimeSeries.from_values(projected_time_series)
  end_time = process_time()
  time = end_time - start_time
  save_time_series_as_csv(input_department,projected_time_series,model.__qualname__,classification.__qualname__,months_to_forecast)
  #plot_scatter(historical_time_series,projected_time_series,input_department,model.__qualname__,classification.__qualname__,months_to_forecast)
  #plot_histogram(historical_time_series,projected_time_series,input_department,model.__qualname__,classification.__qualname__,months_to_forecast)
  save_error(input_department,historical_time_series,projected_time_series,model.__qualname__,classification.__qualname__,months_to_forecast)
  save_time(input_department,time,model.__qualname__,classification.__qualname__,months_to_forecast)
def save_time(
    department:str,
    time:float,
    model:str,
    classification:str,
    months_to_forecast:int
  )->None:
  output_file_name = f'{department}_execution_time.txt'
  path = os.path.join(csv_path,'forecast',classification,model,f'{months_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file = os.path.join(path, output_file_name)
  with open(output_file, 'w') as f:
    f.write(f'{time}')
  print(f"Saved: {output_file}. Elapsed time: {time}")
def save_time_series_as_csv(
    department:str,
    time_series:TimeSeries,
    model:str,
    classification:str,
    months_to_forecast:int
  )->None:
  output_file_name = f'{department}.csv'
  path = os.path.join(csv_path,'forecast',classification,model,f'{months_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file = os.path.join(path, output_file_name)
  time_series.to_csv(output_file, header=False, index=False)
  print(f"Saved: {output_file}")
def plot_scatter(
    actual:TimeSeries,
    predicted:TimeSeries,
    input_department:str,
    model:str,
    classification:str,
    months_to_forecast:int
  )->None:
  plt.figure(figsize=(10, 6))
  plt.scatter(actual,predicted)
  plt.title(f'Scatter plot for {input_department} using {model} - {classification}')
  plt.xlabel('Actual values')
  plt.ylabel('Predicted values')
  plt.legend()
  path = os.path.join(csv_path,'forecast',classification,model,f'{months_to_forecast}_months')
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
  plt.title(f'Histogram for {input_department} using {model} - {classification}')
  plt.xlabel('Error')
  plt.ylabel('Frequency')
  plt.legend()
  path = os.path.join(csv_path,'forecast',classification,model,f'{months_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file_name = f'{input_department}.svg'
  output_file = os.path.join(path, output_file_name)
  plt.savefig(output_file)
  plt.close()
  print(f"Saved: {output_file}")
def save_error(
    input_department:str,
    original_time_series:TimeSeries,
    time_series:TimeSeries,
    model:str,
    classification:str,
    months_to_forecast:int
  )->None:
  path = os.path.join(csv_path,'forecast',classification,model,f'{months_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  mae_error = mae(original_time_series,time_series)
  output_file_name = f'{input_department}_mae.txt'
  output_file = os.path.join(path, output_file_name)
  write_error(output_file,mae_error)
  #mape_error = mape(original_time_series,time_series)
  #output_file_name = f'{input_department}_mape.txt'
  #output_file = os.path.join(path, output_file_name)
  #write_error(output_file,mape_error)
  rmse_error = rmse(original_time_series,time_series)
  output_file_name = f'{input_department}_rmse.txt'
  output_file = os.path.join(path, output_file_name)
  write_error(output_file,rmse_error)
  smape_error = smape(original_time_series,time_series)
  output_file_name = f'{input_department}_smape.txt'
  output_file = os.path.join(path, output_file_name)
  write_error(output_file,smape_error)
def write_error(
    output_file:str,
    error:float
  )->None:
  with open(output_file, 'w') as f:
    f.write(f'{error}')
  print(f"Saved: {output_file}")
#variables
def project_time_series(number_years,
      number_neighbors,
      months_to_forecast,
      classification,
      model
      ):
  for input_department in departments:
    generate_forecast(
      input_department,
      number_years,
      number_neighbors,
      months_to_forecast,
      classification,
      model
      )
number_neighbors=2
number_years=2
months_to_forecast=1
classification=TAN
model=linear_regression
generate_forecast(
      'ASUNCION',
      number_years,
      number_neighbors,
      months_to_forecast,
      classification,
      model
      )
#project_time_series(
#      number_years,
#      number_neighbors,
#      months_to_forecast,
#      classification,
#      model
#      )