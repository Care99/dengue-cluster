from generate.generar_cluster import get_cluster
from generate.generar_cluster_jerarquico import get_cluster_jerarquico
from generate.generar_cluster_de_cluster import get_cluster_de_clusters
from utils.time_series import get_historical_data, get_2022_2023_data
import math
import pandas as pd
import os
from time import time_ns
import matplotlib as mplt
mplt.use('SVG',force=True)
from matplotlib import pyplot as plt
import numpy as np
from darts import concatenate
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers import InvertibleMapper
from models import naive_drift,auto_arima,linear_regression,lstm_forecast
from darts.metrics import mae,rmse,smape
from darts import TimeSeries
import torch
from utils.constants import departments
from plot import plot_scatter,plot_histogram
torch.set_float32_matmul_precision('medium')
#from pmdarima.arima import auto_arima
script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
cluster_matriz_path = os.path.join(csv_path,'cluster_matriz')
matriz_ventana_path = os.path.join(csv_path,'matriz_ventana')
ts_historico_path = os.path.join(csv_path,'ts_historico')
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
def forecast_using_regression_models():
  return 'forecast_using_regression_models'
models = [
  naive_drift,
  auto_arima,
  linear_regression,
  lstm_forecast,
  ]

def generate_forecast(
    input_department,
    number_years,
    number_neighbors,
    weeks_to_forecast,
    classification,
    model):
  #variables
  historical_time_series = get_historical_data(input_department)
  original_time_series = get_2022_2023_data(input_department)
  projected_time_series = []
  for week_index in range(0,53,weeks_to_forecast):
    start_time = time_ns()
    match classification.__qualname__:
      case 'get_historical_data':
        time_series = TimeSeries.from_values(values=historical_time_series)
        scaled_time_series = log_transformer.transform(time_series)
        scaled_forecast = model(time_series,weeks_to_forecast)
        forecasted_values = log_transformer.inverse_transform(scaled_forecast).values().flatten() 
      case 'get_cluster' | 'get_cluster_jerarquico' | 'get_cluster_de_clusters':
        temp_ts = historical_time_series[:-12]
        nearest_neighbors = classification(week_index,input_department,number_years,number_neighbors)
        forecasted_values = []
        for neighbor in nearest_neighbors:
          time_series = []
          time_series.extend(temp_ts)
          time_series.extend(neighbor.values().flatten())
          scaled_time_series = TimeSeries.from_values(values=time_series)
          scaled_time_series = log_transformer.transform(scaled_time_series)
          if(model.__qualname__ == 'lstm_forecast'):
            scaler = Scaler()
            scaled_time_series = scaler.fit_transform(scaled_time_series)
          scaled_forecast = model(scaled_time_series,weeks_to_forecast)
          if(model.__qualname__ == 'lstm_forecast'):
            scaled_forecast = scaler.inverse_transform(scaled_forecast)
          forecast = log_transformer.inverse_transform(scaled_forecast)
          forecasted_values.append(forecast)
        forecasted_values = concatenate(forecasted_values,axis=1).mean(axis=1)
        forecasted_values = forecasted_values.values().flatten()
      case 'CART' | 'RANDOM_FOREST' | 'TAN':
        time_series = TimeSeries.from_values(values=historical_time_series)
        scaled_time_series = log_transformer.transform(time_series)
        scaled_forecast = classification(scaled_time_series,weeks_to_forecast)
        forecasted_values = log_transformer.inverse_transform(scaled_forecast)
        forecasted_values = forecasted_values.values().flatten()
    projected_time_series.extend(forecasted_values)
    historical_time_series.extend(original_time_series[week_index:week_index+weeks_to_forecast])
    historical_time_series = historical_time_series[weeks_to_forecast:]
  historical_time_series = TimeSeries.from_values(values=original_time_series)
  projected_time_series = TimeSeries.from_values(values=projected_time_series[:53])
  end_time = time_ns()
  time = float(end_time - start_time) / float(1e9)
  save_time_series_as_csv(input_department,projected_time_series,model.__qualname__,classification.__qualname__,weeks_to_forecast)
  plot_scatter(historical_time_series,projected_time_series,input_department,model.__qualname__,classification.__qualname__,weeks_to_forecast)
  plot_histogram(historical_time_series,projected_time_series,input_department,model.__qualname__,classification.__qualname__,weeks_to_forecast)
  save_error(input_department,historical_time_series,projected_time_series,model.__qualname__,classification.__qualname__,weeks_to_forecast)
  save_time(input_department,time,model.__qualname__,classification.__qualname__,weeks_to_forecast)
def save_time(
    department:str,
    time:float,
    model:str,
    classification:str,
    weeks_to_forecast:int
  )->None:
  output_file_name = f'{department}_execution_time.txt'
  path = os.path.join(csv_path,'forecast',classification,model,f'{weeks_to_forecast}_months')
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
    weeks_to_forecast:int
  )->None:
  output_file_name = f'{department}.csv'
  path = os.path.join(csv_path,'forecast',classification,model,f'{weeks_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file = os.path.join(path, output_file_name)
  time_series.to_csv(output_file, header=False, index=False)
  print(f"Saved: {output_file}")
def save_error(
    input_department:str,
    original_time_series:TimeSeries,
    time_series:TimeSeries,
    model:str,
    classification:str,
    weeks_to_forecast:int
  )->None:
  path = os.path.join(csv_path,'forecast',classification,model,f'{weeks_to_forecast}_months')
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
  try:
    smape_error = smape(original_time_series,time_series)
  except ValueError as e:
    print(f"Darts SMAPE failed: {e}")
    print("Calculating SMAPE manually...")
    
    # Extract values
    original_vals = original_time_series.values().flatten()
    predicted_vals = time_series.values().flatten()
    
    # Ensure arrays have same length
    min_len = min(len(original_vals), len(predicted_vals))
    original_vals = original_vals[:min_len]
    predicted_vals = predicted_vals[:min_len]
    
    # Use absolute values to ensure positivity
    abs_original = np.abs(original_vals)
    abs_predicted = np.abs(predicted_vals)
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-10
    abs_original = np.where(abs_original == 0, epsilon, abs_original)
    abs_predicted = np.where(abs_predicted == 0, epsilon, abs_predicted)
    
    # Calculate SMAPE manually
    numerator = np.abs(abs_original - abs_predicted)
    denominator = abs_original + abs_predicted
    smape_vals = 100 * numerator / denominator
    smape_error = np.mean(smape_vals)
    
    print(f"SMAPE (manual calculation with absolute values): {smape_error:.2f}%")
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
      weeks_to_forecast,
      classification,
      model
      ):
  for input_department in departments:
    generate_forecast(
      input_department,
      number_years,
      number_neighbors,
      weeks_to_forecast,
      classification,
      model
      )
number_years = 2
number_neighbors = 2
#for classification in [CART,RANDOM_FOREST]:
#    for weeks_to_forecast in [1,2,3,4]:
#      for input_department in departments:
#        generate_forecast(
#          input_department,
#          number_years,
#          number_neighbors,
#          weeks_to_forecast,
#          classification,
#          forecast_using_regression_models
#        )
#for classification in [get_historical_data,get_cluster,get_cluster_jerarquico,get_cluster_de_clusters]:
#  for model in models:
#    for weeks_to_forecast in [1,2,3,4]:
#      for input_department in departments:
#        generate_forecast(
#          input_department,
#          number_years,
#          number_neighbors,
#          weeks_to_forecast,
#          classification,
#          model
#        )
