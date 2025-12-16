from src.classifiers import CART,RANDOM_FOREST
from src.forecast import generate_forecast,forecast_using_regression_models
from src.generate.generar_cluster import get_cluster
from src.generate.generar_cluster_jerarquico import get_cluster_jerarquico
from src.generate.generar_cluster_de_cluster import get_cluster_de_clusters
from src.models import naive_drift,auto_arima,linear_regression,lstm_forecast
from src.utils.constants import departments
from src.utils.time_series import get_historical_data
models = [
  naive_drift,
  auto_arima,
  linear_regression,
  lstm_forecast,
  ]
number_years = 2
number_neighbors = 2
for classification in [CART,RANDOM_FOREST]:
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