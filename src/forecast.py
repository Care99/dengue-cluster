from src.output import save_error,save_time,save_time_series_as_csv
from src.plot import plot_scatter,plot_histogram
from src.utils.constants import departments
from src.utils.time_series import get_historical_data, get_2022_2023_data

import datetime as dt
from darts import concatenate, TimeSeries
from darts.dataprocessing.transformers import Scaler, InvertibleMapper
import numpy as np

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
  start_time = dt.datetime.now()
  for week_index in range(0,53,weeks_to_forecast):
    forecasted_values: list[float] = []
    match classification.__qualname__:
      case 'get_historical_data':
        time_series = TimeSeries.from_values(values=np.array(historical_time_series))
        scaled_time_series = log_transformer.transform(time_series)
        start_time = dt.datetime.now()
        scaled_forecast: TimeSeries = model(time_series,weeks_to_forecast)
        forecasted_values = log_transformer.inverse_transform(series=scaled_forecast).values().flatten().tolist() 
      case 'get_cluster' | 'get_cluster_jerarquico' | 'get_cluster_de_clusters':
        temp_ts = historical_time_series[:-12]
        nearest_neighbors: list[TimeSeries] = classification(week_index,input_department,number_years,number_neighbors)
        list_forecasted_time_series: list[TimeSeries] = []
        for neighbor in nearest_neighbors:
          time_series = temp_ts + neighbor.values().flatten().tolist()
          scaled_time_series = TimeSeries.from_values(values=np.array(time_series))
          scaled_time_series = log_transformer.transform(scaled_time_series)
          scaler = Scaler()
          if(model.__qualname__ == 'lstm_forecast'):
            scaled_time_series = scaler.fit_transform(scaled_time_series)
          start_time = dt.datetime.now()
          scaled_forecast: TimeSeries = model(scaled_time_series,weeks_to_forecast)
          if(model.__qualname__ == 'lstm_forecast'):
            scaled_forecast = scaler.inverse_transform(scaled_forecast)
          forecast: TimeSeries = log_transformer.inverse_transform(scaled_forecast)
          list_forecasted_time_series.append(forecast)
        forecasted_values = concatenate(list_forecasted_time_series,axis=1).mean(axis=1).values().flatten().tolist()
      case 'CART' | 'RANDOM_FOREST' | 'TAN':
        time_series = TimeSeries.from_values(values=np.array(historical_time_series))
        scaled_time_series = log_transformer.transform(time_series)
        start_time = dt.datetime.now()
        scaled_forecast: TimeSeries = classification(scaled_time_series,weeks_to_forecast)
        forecasted_values = log_transformer.inverse_transform(scaled_forecast).values().flatten().tolist()
    projected_time_series.extend(forecasted_values)
    historical_time_series.extend(original_time_series[week_index:week_index+weeks_to_forecast])
    historical_time_series = historical_time_series[weeks_to_forecast:]
  expected_time_series = TimeSeries.from_values(values=np.array(original_time_series))
  observed_time_series = TimeSeries.from_values(values=np.array(projected_time_series[:53]))
  end_time = dt.datetime.now()
  # calculate elapsed time in seconds
  time = end_time.timestamp() - start_time.timestamp()
  save_time_series_as_csv(input_department,observed_time_series,model.__qualname__,classification.__qualname__,weeks_to_forecast)
  plot_scatter(expected_time_series,observed_time_series,input_department,model.__qualname__,classification.__qualname__,weeks_to_forecast)
  plot_histogram(expected_time_series,observed_time_series,input_department,model.__qualname__,classification.__qualname__,weeks_to_forecast)
  save_error(input_department,expected_time_series,observed_time_series,model.__qualname__,classification.__qualname__,weeks_to_forecast)
  save_time(input_department,time,model.__qualname__,classification.__qualname__,weeks_to_forecast)
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
