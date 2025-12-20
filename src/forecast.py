from src.output import save_error,save_time,save_time_series_as_csv
from src.plot import plot_scatter,plot_histogram
from src.utils.constants import departments,time_series_window,time_series_2022_2023_length
from src.utils.time_series import get_historical_data, get_2022_2023_data, get_historical_data_window, get_time_series_window

import datetime as dt
from darts import concatenate, TimeSeries
from darts.models import AutoARIMA,LinearRegressionModel,NaiveDrift,RNNModel,SKLearnModel
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Apply log transformation (ensure all values > 0)
def safe_log(x:TimeSeries)->TimeSeries:
    return TimeSeries.from_values(values=np.log1p(x.to_dataframe().to_numpy()))  # log(1 + x) handles zeros

def safe_exp(x:TimeSeries)->TimeSeries:
    return TimeSeries.from_values(values=np.expm1(x.to_dataframe().to_numpy()))  # exp(x) - 1 inverse

def forecast_using_regression_models():
  return 'forecast_using_regression_models'

def forecast(
    time_series:TimeSeries,
    forecast_values:int,
    model,
  )->list[float]:
  scaled_time_series:TimeSeries = safe_log(time_series)
  scaler = MinMaxScaler()
  forecast_model = model()
  if(model.__qualname__=="lstm_model"):
    scaled_time_series = TimeSeries.from_values(values=scaler.fit_transform(scaled_time_series.to_dataframe().to_numpy()))
  forecast_model.fit(scaled_time_series)
  generated_scaled_time_series: TimeSeries = forecast_model.predict(forecast_values)
  if(model.__qualname__=="lstm_model"):
    generated_scaled_time_series = TimeSeries.from_values(scaler.inverse_transform(generated_scaled_time_series.to_dataframe().to_numpy()))
  generated_time_series = safe_exp(generated_scaled_time_series)
  return generated_time_series.values().flatten().tolist()

def generate_state_of_art_forecast(
    input_department:str,
    weeks_to_forecast:int,
    classification,
    week_index:int
  ):
  #variables
  historical_time_series: list[float] = get_historical_data_window(input_department,week_index)
  time_series = TimeSeries.from_values(values=np.array(historical_time_series))
  return forecast(time_series,weeks_to_forecast,classification)
  
def generate_historical_data_forecast(
    input_department:str,
    weeks_to_forecast:int,
    model,
    week_index:int
  ):
  #variables
  historical_time_series: list[float] = get_historical_data_window(input_department,week_index)
  time_series = TimeSeries.from_values(values=np.array(historical_time_series))
  return forecast(time_series,weeks_to_forecast,model)

def generate_cluster_forecast(
    input_department:str,
    number_years:int,
    number_neighbors:int,
    weeks_to_forecast:int,
    classification,
    model,
    week_index:int
  ):
  #variables
  historical_time_series: list[float] = get_time_series_window(input_department,week_index)
  nearest_neighbors: list[list[float]] = classification(week_index,input_department,number_years,number_neighbors)
  for i in range(len(nearest_neighbors)):
    nearest_neighbors[i] = historical_time_series + nearest_neighbors[i]
  list_forecasted_time_series: list[TimeSeries] = [TimeSeries.from_values(np.array(forecast(TimeSeries.from_values(neighbor),weeks_to_forecast,model)))for neighbor in nearest_neighbors]
  return concatenate(list_forecasted_time_series,axis=1).mean(axis=1).values().flatten().tolist()

def save_to_disk(
    input_department:str,
    expected_time_series:TimeSeries,
    observed_time_series:TimeSeries,
    model_name:str,
    classification_name:str,
    weeks_to_forecast:int,
  ):
  save_time_series_as_csv(input_department,observed_time_series,model_name,classification_name,weeks_to_forecast)
  plot_scatter(expected_time_series,observed_time_series,input_department,model_name,classification_name,weeks_to_forecast)
  plot_histogram(expected_time_series,observed_time_series,input_department,model_name,classification_name,weeks_to_forecast)
  save_error(input_department,expected_time_series,observed_time_series,model_name,classification_name,weeks_to_forecast)
