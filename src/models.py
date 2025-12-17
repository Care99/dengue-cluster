#Forecasting models
#Baseline Models 
from darts import TimeSeries
from darts.models import NaiveDrift
#Statistical Models 
from darts.models import AutoARIMA
#SKLearn-Like Models 
from darts.models import LinearRegressionModel
#PyTorch (Lightning)-based Models
from darts.models import RNNModel
import torch

torch.set_float32_matmul_precision('medium')
def naive_drift(time_series:TimeSeries,forecasted_values:int)->TimeSeries:
  data = time_series
  model = NaiveDrift()
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series

def auto_arima(time_series:TimeSeries,forecasted_values:int)->TimeSeries:
  data = time_series
  #train, test = model_selection.train_test_split(data)
  model = AutoARIMA(season_length=4)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series
def linear_regression(time_series:TimeSeries,forecasted_values:int)->TimeSeries:
  data = time_series
  model = LinearRegressionModel(lags=12)
  model.fit(data)
  generated_time_series = model.predict(forecasted_values)
  return generated_time_series
def lstm_forecast(time_series:TimeSeries,forecasted_values:int)->TimeSeries:
  data = time_series
  model = RNNModel(
    model='LSTM',
    input_chunk_length=12,
    output_chunk_length=forecasted_values,
    hidden_dim=25, 
    n_rnn_layers=1, 
    dropout=0.0, 
    batch_size=52, 
    n_epochs=100,
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