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
    forecasted_values: list[float] = []
    match classification_name:
      case 'get_historical_data':
        time_series = TimeSeries.from_values(values=np.array(historical_time_series))
        forecasted_values = model(time_series,weeks_to_forecast)
      case 'get_cluster' | 'get_cluster_jerarquico' | 'get_cluster_de_clusters':
        temp_ts = historical_time_series[:-time_series_window]
        nearest_neighbors: list[TimeSeries] = classification(week_index,input_department,number_years,number_neighbors)
        list_forecasted_time_series: list[TimeSeries] = []
        for neighbor in nearest_neighbors:
          np_time_series = temp_ts + neighbor.values().flatten().tolist()
          time_series = TimeSeries.from_values(values=np.array(np_time_series))
          scaler = MinMaxScaler()
          if(model_name == 'lstm_forecast'):
            time_series = TimeSeries.from_values(values=scaler.fit_transform(time_series.to_dataframe().to_numpy()))
          forecast: TimeSeries = TimeSeries.from_values(np.array(model(time_series,weeks_to_forecast)))
          if(model_name == 'lstm_forecast'):
            forecast = TimeSeries.from_values(scaler.inverse_transform(forecast.to_dataframe().to_numpy()))
          list_forecasted_time_series.append(forecast)
        forecasted_values = concatenate(list_forecasted_time_series,axis=1).mean(axis=1).values().flatten().tolist()
      case 'CART' | 'RANDOM_FOREST' | 'TAN':
        time_series = TimeSeries.from_values(values=np.array(historical_time_series))
        forecasted_values = classification(time_series,weeks_to_forecast)
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
