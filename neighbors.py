from generar_cluster import get_cluster
from generar_cluster_jerarquico import get_cluster_jerarquico
from generar_cluster_de_cluster import get_cluster_de_clusters
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
def forecast_using_regression_models():
  return 'forecast_using_regression_models'
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

def get_2022_2023_data(input_department):
  historical_time_series = []
  for year in range(2022,2024):
    match year:
      case 2022:
        for month in ["OCTUBRE","NOVIEMBRE","DICIEMBRE"]:
          filename = f'{input_department}.csv'
          path = os.path.join(ts_historico_path,str(year),month,filename)
          temp_ts = load_time_series(path)
          for value in temp_ts:
            historical_time_series.append(value)
      case 2023:
        for month in ["ENERO","FEBRERO","MARZO","ABRIL","MAYO","JUNIO","JULIO","AGOSTO","SEPTIEMBRE"]:
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
def plot_scatter(
    actual:TimeSeries,
    predicted:TimeSeries,
    input_department:str,
    model:str,
    classification:str,
    weeks_to_forecast:int
  )->None:
  actual = actual.values().flatten()
  predicted = predicted.values().flatten()
  plt.figure(figsize=(10, 6))
  # Calculate regression line for reference
  z = np.polyfit(actual, predicted, 1)
  p = np.poly1d(z)
  x_line = np.linspace(min(actual), max(actual), 100)
  y_line = p(x_line)
  
  # Create color gradient based on prediction error
  errors = np.abs(predicted - actual)
  normalized_errors = (errors - errors.min()) / (errors.max() - errors.min())
  plt.scatter(
    actual,
    predicted,
    c=normalized_errors,
    cmap='viridis',
    alpha=0.7,
    s=50 + 100 * normalized_errors,  # Size varies with error
  )
  # Perfect prediction line (y = x)
  min_val = actual.min()
  max_val = actual.max()
  plt.plot(
    [min_val, max_val], 
    [min_val, max_val], 
    'r--', 
    alpha=0.8, 
    linewidth=2
  )
  # Regression line
  plt.plot(
    x_line, 
    y_line, 
    'orange', 
    alpha=0.8, 
    linewidth=2, 
    label=f'Regression (slope={z[0]:.3f})'
  )    
  # Title and labels with better formatting
  plt.title(
      f'Scatter Plot: {input_department}\n'
      f'Model: {model} | Classification: {classification} | Horizon: {weeks_to_forecast} months',
      fontsize=14, fontweight='bold', pad=20
  )
  plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
  plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
  r2 = np.corrcoef(actual, predicted)[0, 1]**2
  plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
  plt.legend()
  path = os.path.join(csv_path,'forecast',classification,model,f'{weeks_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file_name = f'{input_department}_scatter.svg'
  output_file = os.path.join(path, output_file_name)
  plt.savefig(output_file)
  plt.close()
  print(f"Saved: {output_file}")
def plot_histogram(
    actual: TimeSeries,
    predicted: TimeSeries,
    input_department: str,
    model: str,
    classification: str,
    weeks_to_forecast: int
) -> None:
    # Extract data
    actual_vals = actual.values().flatten()
    predicted_vals = predicted.values().flatten()
    
    # Calculate IQR for actual values to determine reasonable bounds
    actual_q1 = np.percentile(actual_vals, 25)
    actual_q3 = np.percentile(actual_vals, 75)
    actual_iqr = actual_q3 - actual_q1
    
    # Define bounds based on actual values (using 1.5*IQR rule or actual min/max)
    lower_bound = max(predicted_vals.min(), actual_vals.min())
    upper_bound = min(predicted_vals.max(), actual_q3 + 1.5 * actual_iqr)
    
    # Filter predicted values for visualization (but keep all for stats)
    predicted_filtered = predicted_vals[(predicted_vals >= lower_bound) & 
                                       (predicted_vals <= upper_bound)]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    # Use bins based on actual values range
    bins = np.histogram_bin_edges(actual_vals, bins='auto')
    
    # --- Plot 1: Zoomed-in view (using actual value range) ---
    ax1.hist(predicted_vals, bins=bins, alpha=0.6, label='Predicted',
             color='#FF6B6B', edgecolor='black', linewidth=0.5,
             range=(lower_bound, upper_bound))
    ax1.hist(actual_vals, bins=bins, alpha=0.6, label='Actual',
             color='#4ECDC4', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Zoomed View (Actual Value Range)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add outlier info
    outlier_count = len(predicted_vals) - len(predicted_filtered)
    outlier_pct = (outlier_count / len(predicted_vals)) * 100
    info_text = f'Focus Range: [{lower_bound:.1f}, {upper_bound:.1f}]\n' \
                f'Outliers excluded: {outlier_count} ({outlier_pct:.1f}%)'
    ax1.text(0.08, 0.98, info_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # --- Plot 2: Box plot for outlier visualization ---
    data_to_plot = [actual_vals, predicted_vals]
    box_colors = ['#4ECDC4', '#FF6B6B']
    
    bp = ax2.boxplot(data_to_plot, patch_artist=True, labels=['Actual', 'Predicted'],
                     showfliers=True, whis=1.5)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Color the medians
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    
    # Add individual points for actual values (in background)
    for i, (data, color) in enumerate(zip(data_to_plot, box_colors), 1):
        # Add jitter to x-coordinate
        x = np.random.normal(i, 0.04, size=len(data))
        ax2.scatter(x, data, alpha=0.3, color=color, s=20)
    
    ax2.set_ylabel('Box Plot with Outliers', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add statistics as text
    actual_stats = f'Actual: μ={np.mean(actual_vals):.1f}, σ={np.std(actual_vals):.1f}'
    pred_stats = f'Predicted: μ={np.mean(predicted_vals):.1f}, σ={np.std(predicted_vals):.1f}'
    ax2.text(0.98, 0.98, f'{actual_stats}\n{pred_stats}', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Main title
    fig.suptitle(
        f'Distribution Analysis with Outlier Handling: {input_department}\n'
        f'Model: {model} | Classification: {classification} | Horizon: {weeks_to_forecast} months',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    # Save
    path = os.path.join(csv_path, 'forecast', classification, model, f'{weeks_to_forecast}_months')
    os.makedirs(path, exist_ok=True)
    output_file_name = f'{input_department}_hist.svg'
    output_file = os.path.join(path, output_file_name)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")
    print(f"Outlier info: {outlier_count} predicted values ({outlier_pct:.1f}%) excluded from histogram view")
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
