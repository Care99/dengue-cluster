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
def naive_drift_model()->NaiveDrift:
  return NaiveDrift()
def auto_arima_model()->AutoARIMA:
  return AutoARIMA(season_length=52)
def linear_regression_model()->LinearRegressionModel:
  return LinearRegressionModel(lags=12)
def lstm_model()->RNNModel:
  return RNNModel(
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