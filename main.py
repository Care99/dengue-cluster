from src.classifiers import classifiers as Classifiers
from src.forecast import generate_state_of_art_forecast,generate_historical_data_forecast,generate_cluster_forecast,save_to_disk
from src.generate.generar_cluster import get_cluster
from src.generate.generar_cluster_jerarquico import get_cluster_jerarquico
from src.generate.generar_cluster_de_cluster import get_cluster_de_clusters
from src.models import models as Models
from src.output import save_time
from src.utils.constants import departments,time_series_2022_2023_length
from src.utils.time_series import get_historical_data,get_2022_2023_data

from darts import TimeSeries
import datetime as dt
import numpy as np
md = Models()
clsf = Classifiers()
models = [
  [md.naive_drift_model,"NAIVE_MODEL"],
  [md.auto_arima_model,"AUTO_ARIMA"],
  [md.linear_regression_model,"LINEAR_REGRESSION"],
  [md.lstm_model,"LSTM"],
]
number_years = 2
number_neighbors = 2
def state_of_art():
  # Forecast CART and RANDOM_FOREST
  classifications = [
    [clsf.cart_model,"CART"],
    [clsf.rf_model,"RANDOM_FOREST"]
  ]
  for classification in classifications:
    start_time = dt.datetime.now()
    for weeks_to_forecast in [1,2,3,4]:
      for input_department in departments:
        original_time_series: list[float] = get_2022_2023_data(input_department)
        projected_time_series: list[float] = []
        for week_index in range(0,time_series_2022_2023_length,weeks_to_forecast):
          forecasted_values:list[float]=generate_state_of_art_forecast(input_department,weeks_to_forecast,classification[0],week_index)
          projected_time_series.extend(forecasted_values)
        expected_time_series = TimeSeries.from_values(values=np.array(original_time_series))
        observed_time_series = TimeSeries.from_values(values=np.array(projected_time_series[:time_series_2022_2023_length]))
        save_to_disk(input_department,expected_time_series,observed_time_series,"state_of_art",classification[1],weeks_to_forecast)
    end_time = dt.datetime.now()
    # calculate elapsed time in seconds
    time = end_time.timestamp() - start_time.timestamp()
    save_time(time,"state_of_art",classification[1])

def historical_data():
  # Foreacst Historical Data
  for model in models:
    start_time = dt.datetime.now()
    for weeks_to_forecast in [1,2,3,4]:
      for input_department in departments:
        original_time_series: list[float] = get_2022_2023_data(input_department)
        projected_time_series: list[float] = []
        for week_index in range(0,time_series_2022_2023_length,weeks_to_forecast):
          forecasted_values:list[float]=generate_historical_data_forecast(input_department,weeks_to_forecast,model[0],week_index)
          projected_time_series.extend(forecasted_values)
        expected_time_series = TimeSeries.from_values(values=np.array(original_time_series))
        observed_time_series = TimeSeries.from_values(values=np.array(projected_time_series[:time_series_2022_2023_length]))
        save_to_disk(input_department,expected_time_series,observed_time_series,model[1],"HISTORICAL_DATA",weeks_to_forecast)
    end_time = dt.datetime.now()
    time = end_time.timestamp() - start_time.timestamp()
    save_time(time,model[1],'historical_data')

def cluster():
  # Forecast with clusters
  for classification in [get_cluster,get_cluster_jerarquico,get_cluster_de_clusters]:
    for model in models:
      start_time = dt.datetime.now()
      for weeks_to_forecast in [1,2,3,4]:
        for input_department in departments:
          original_time_series: list[float] = get_2022_2023_data(input_department)
          projected_time_series: list[float] = []
          for week_index in range(0,time_series_2022_2023_length,weeks_to_forecast):
            forecasted_values:list[float]=generate_cluster_forecast(
              input_department,
              number_years,
              number_neighbors,
              weeks_to_forecast,
              classification,
              model[0],
              week_index
            )
            projected_time_series.extend(forecasted_values)
          expected_time_series = TimeSeries.from_values(values=np.array(original_time_series))
          observed_time_series = TimeSeries.from_values(values=np.array(projected_time_series[:time_series_2022_2023_length]))
          save_to_disk(input_department,expected_time_series,observed_time_series,model.__qualname__,"historical_data",weeks_to_forecast)
      end_time = dt.datetime.now()
      time = end_time.timestamp() - start_time.timestamp()
      save_time(time,model.__qualname__,'historical_data')

state_of_art()
historical_data()
cluster()