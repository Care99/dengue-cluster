from generar_cluster import get_k_n_n as c_get_knn
from generar_cluster_jerarquico import get_k_n_n as cj_get_knn
from generar_cluster_de_cluster import get_k_n_n as cdc_get_knn
import math
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.spatial.distance import euclidean

#from tslearn.metrics import dtw
import os
from time import process_time
import matplotlib as mplt

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Theta, HoltWinters, Naive, SeasonalNaive, CrostonClassic, CrostonOptimized


from statsforecast import StatsForecast
import pandas as pd
import numpy as np

script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
cluster_matriz_path = os.path.join(csv_path,'cluster_matriz')
matriz_ventana_path = os.path.join(csv_path,'matriz_ventana')
ts_historico_path = os.path.join(csv_path,'ts_historico')


funciones_cluster = [c_get_knn, cj_get_knn, cdc_get_knn]
nombre_funciones_cluster = ["cluster", "cluster_jerarquico", "cluster_de_cluster"]

BASE_PATH_HISTORICO = "csv/ts_historico"


departments = [
  'ALTO_PARANA',
  'AMAMBAY',
  'ASUNCION'
  'CAAGUAZU',
  'CENTRAL',
  'CENTRO_EST',
  'CENTRO_NORTE',
  'CENTRO_SUR',
  'CHACO',
  'CORDILLERA',
  'METROPOLITANO',
  'PARAGUARI',
  'PARAGUAY',
  'PTE_HAYES',
  'SAN_PEDRO',
  'CANINDEYU',
  'CONCEPCION',
  'ITAPUA',
  'MISIONES',
  'BOQUERON',
  'GUAIRA',
  'CAAZAPA',
  'NEEMBUCU',
  'ALTO_PARAGUAY'
  ]
conjunto_funciones = [
   "bhattacharyya",
]
month_year_list_prediction=[
    ['OCTUBRE','2022'],
    ['NOVIEMBRE','2022'],
    ['DICIEMBRE','2022'],
    ['ENERO','2023'],
    ['FEBRERO','2023'],
    ['MARZO','2023'],
    ['ABRIL','2023'],
    ['MAYO','2023'],
    ['JUNIO','2023'],
    ['JULIO','2023'],
    ['AGOSTO','2023'],
    ['SEPTIEMBRE','2023']
  ]

month_map = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12
}


def forecast_1_month(ts, month_year, horizon):
    # Ensure ts is 1D numpy array
    ts_values = np.array(ts).flatten()
    month_num = month_map[month_year[0].upper()]
    start_date = pd.Timestamp(year=int(month_year[1]), month=month_num, day=1)

    # Build DataFrame for StatsForecast
    df = pd.DataFrame({
        "unique_id": ["ts"] * len(ts_values),
        "ds": pd.date_range(start=start_date, periods=len(ts_values), freq="W"),
        "y": ts_values
    })

    # Define models
    models = [
        # Limit differencing to avoid warnings on short series
        AutoARIMA(max_d=1, max_D=1, seasonal=True, season_length=4, start_p=0, max_p=2, start_q=0, max_q=2),
        Theta(),
        HoltWinters(season_length=4),
        Naive(),
        SeasonalNaive(season_length=4),
        CrostonClassic(),
        CrostonOptimized()
    ]

    # Initialize forecast dictionary with NaNs
    forecast_dict = {type(m).__name__: np.full(horizon, np.nan) for m in models}

    # Fit and predict safely
    try:
        sf = StatsForecast(models=models, freq="W", n_jobs=1)
        sf.fit(df)
        fcst = sf.predict(h=horizon)
        # Fill forecast_dict with actual values
        for model_name in forecast_dict.keys():
            if model_name in fcst.columns:
                forecast_dict[model_name] = fcst[model_name].values
    except Exception as e:
        print(f"Forecast failed for {month_year}: {e}")

    return forecast_dict

def get_real_data_ts(department,mes,year):
        csv_path = os.path.join(BASE_PATH_HISTORICO, str(year), mes, f"{department}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} does not exist")    
        # CSV has no header, assume single column
        df = pd.read_csv(csv_path, header=None)
        return pd.Series(df.values.flatten())


def generar_serie_tiempo(deps,month_year_list):

    results = {}  # Nested dictionary to store forecasts
    for dep in deps:
        results[dep] = {}

        for month, year in month_year_list:
            results[dep][month] = {}
            real_data_ts = get_real_data_ts(dep,month,year)
            print(f"datos real: {real_data_ts}, size:{len(real_data_ts)}")
            for fc_idx, fc in enumerate(funciones_cluster):
                funcion_name = fc.__name__

                # Get nearest neighbors from cluster function
                ts_neighbors = fc(month, dep, 4, 2)

                # Initialize dictionary automatically using the first neighbor
                if ts_neighbors:
                    first_forecast = forecast_1_month(ts_neighbors[0].values().flatten(), (month, year), len(real_data_ts) )
                    forecasts_each_ts = {model_name: [] for model_name in first_forecast.keys()}
                else:
                    forecasts_each_ts = {}

                # Collect forecasts for all neighbors
                for ts in ts_neighbors:
                    ts_values = ts.values().flatten()
                    f = forecast_1_month(ts_values, (month, year),len(real_data_ts))
                    for model_name, forecast in f.items():
                        forecasts_each_ts[model_name].append(forecast)

                # Compute average forecast across neighbors
                avg_forecasts = {
                    model_name: np.mean(np.stack(arrays), axis=0)
                    for model_name, arrays in forecasts_each_ts.items()
                }

                results[dep][month][funcion_name] = avg_forecasts

                # Print in clean, automatic format
                print(f"metodo cluster: {nombre_funciones_cluster[fc_idx]}, "
                    f"mes: {month}, año: {year}, departamento: {dep}, función: {funcion_name}:")
                for model_name, forecast_array in avg_forecasts.items():
                    print(f"\t{model_name}: {forecast_array}")
                print()  # blank line between clusters

department_test = ['CENTRAL']
month_year_list_test=[
        ['JULIO','2023']
    ]


generar_serie_tiempo(department_test,month_year_list_test)