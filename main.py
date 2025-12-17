from src.classifiers import classifiers
from src.forecast import generate_forecast,forecast_using_regression_models
from src.generate.generar_cluster import get_cluster
from src.generate.generar_cluster_jerarquico import get_cluster_jerarquico
from src.generate.generar_cluster_de_cluster import get_cluster_de_clusters
from src.models import models
from src.utils.constants import departments
from src.utils.time_series import get_historical_data
md = models()
clsf = classifiers()
models = [
  md.naive_drift,
  md.auto_arima,
  md.linear_regression,
  md.lstm_forecast,
  ]
number_years = 2
number_neighbors = 2
for classification in [clsf.CART,clsf.RANDOM_FOREST]:
    for weeks_to_forecast in [1,2,3,4]:
      for input_department in departments:
        generate_forecast(
          input_department,
          number_years,
          number_neighbors,
          weeks_to_forecast,
          classification,
          forecast_using_regression_models
        )
for classification in [get_historical_data,get_cluster,get_cluster_jerarquico,get_cluster_de_clusters]:
  for model in models:
    for weeks_to_forecast in [1,2,3,4]:
      for input_department in departments:
        generate_forecast(
          input_department,
          number_years,
          number_neighbors,
          weeks_to_forecast,
          classification,
          model
        )