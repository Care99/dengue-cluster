from src.utils.constants import csv_path

from darts import TimeSeries
from darts.metrics import mae,rmse,smape
import numpy as np
import os
def save_time(
    time:float,
    model:str,
    classification:str,
    )->None:
  output_file_name = f'execution_time.txt'
  path = os.path.join(csv_path,'forecast',classification,model)
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