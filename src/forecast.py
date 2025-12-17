from src.output import save_error,save_time,save_time_series_as_csv
from src.plot import plot_scatter,plot_histogram
from src.utils.constants import departments,time_series_window,time_series_2022_2023_length
from src.utils.time_series import get_historical_data, get_2022_2023_data

import datetime as dt
from darts import concatenate, TimeSeries
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
    classifier
  )->list[float]:
  model_name = model.__qualname__
  scaled_time_series:TimeSeries = safe_log(time_series)
  if(model_name==forecast_using_regression_models.__qualname__):
    forecast_model = classifier()
  else:
    forecast_model = model()
  scaler = MinMaxScaler()
  if(model_name == 'lstm_forecast'):
    scaled_time_series = TimeSeries.from_values(values=scaler.fit_transform(scaled_time_series.to_dataframe().to_numpy()))
  forecast_model.fit(scaled_time_series)
  generated_scaled_time_series: TimeSeries = forecast_model.predict(forecast_values)
  if(model_name == 'lstm_forecast'):
    generated_scaled_time_series = TimeSeries.from_values(scaler.inverse_transform(generated_scaled_time_series.to_dataframe().to_numpy()))
  generated_time_series = safe_exp(generated_scaled_time_series)
  return generated_time_series.values().flatten().tolist()

def generate_forecast(
    input_department:str,
    number_years:int,
    number_neighbors:int,
    weeks_to_forecast:int,
    classification,
    model):
  #variables
  historical_time_series: list[float] = get_historical_data(input_department)
  original_time_series: list[float] = get_2022_2023_data(input_department)
  projected_time_series: list[float] = []
  model_name = model.__qualname__
  classification_name = classification.__qualname__
  start_time = dt.datetime.now()
  for week_index in range(0,time_series_2022_2023_length,weeks_to_forecast):
    if (classification_name in ['get_cluster','get_cluster_jerarquico','get_cluster_de_clusters']):
      nearest_neighbors: list[list[float]] = classification(week_index,input_department,number_years,number_neighbors)
      for i in range(len(nearest_neighbors)):
        nearest_neighbors[i] = historical_time_series[:-time_series_window] + nearest_neighbors[i]
      list_forecasted_time_series: list[TimeSeries] = [TimeSeries.from_values(np.array(forecast(TimeSeries.from_values(neighbor),weeks_to_forecast,model,classification)))for neighbor in nearest_neighbors]
      forecasted_values: list[float] = concatenate(list_forecasted_time_series,axis=1).mean(axis=1).values().flatten().tolist()
    else:
      time_series = TimeSeries.from_values(values=np.array(historical_time_series))
      forecasted_values = forecast(time_series,weeks_to_forecast,model,classification)
    projected_time_series.extend(forecasted_values)
    historical_time_series.extend(original_time_series[week_index:week_index+weeks_to_forecast])
    historical_time_series = historical_time_series[weeks_to_forecast:]
  expected_time_series = TimeSeries.from_values(values=np.array(original_time_series))
  observed_time_series = TimeSeries.from_values(values=np.array(projected_time_series[:time_series_2022_2023_length]))
  end_time = dt.datetime.now()
  # calculate elapsed time in seconds
  time = end_time.timestamp() - start_time.timestamp()
  save_time_series_as_csv(input_department,observed_time_series,model_name,classification_name,weeks_to_forecast)
  plot_scatter(expected_time_series,observed_time_series,input_department,model_name,classification_name,weeks_to_forecast)
  plot_histogram(expected_time_series,observed_time_series,input_department,model_name,classification_name,weeks_to_forecast)
  save_error(input_department,expected_time_series,observed_time_series,model_name,classification_name,weeks_to_forecast)
  save_time(input_department,time,model_name,classification_name,weeks_to_forecast)
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
