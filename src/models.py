from src.forecast import safe_log,safe_exp
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
class models:
  def __init__(self):
    self.naive_drift_model = NaiveDrift()
    self.auto_arima_model = AutoARIMA(season_length=52)
    self.linear_regression_model = LinearRegressionModel(lags=12)
    self.lstm_model = RNNModel(
        model='LSTM',
        input_chunk_length=12,
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

  def naive_drift(self)->NaiveDrift:
    return self.naive_drift_model

  def auto_arima(self)->AutoARIMA:
    return self.auto_arima_model

  def linear_regression(self)->LinearRegressionModel:
    return self.linear_regression_model

  def lstm_forecast(self)->RNNModel:
    return self.lstm_model