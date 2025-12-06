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
from statsforecast.models import (
    Theta,
    AutoETS,
    CrostonOptimized,
    CrostonClassic
)

from statsforecast import StatsForecast
import pandas as pd
import numpy as np

script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
cluster_matriz_path = os.path.join(csv_path,'cluster_matriz')
matriz_ventana_path = os.path.join(csv_path,'matriz_ventana')
ts_historico_path = os.path.join(csv_path,'ts_historico')


funciones_cluster = [c_get_knn, cj_get_knn, cdc_get_knn]

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

department_test = ['BOQUERON']
month_year_list_test=[
        ['ENERO','2023']
    ]


def forecast_1_month(ts, month_year, h=4):
    # Convert ts to 1D array if it isn't
    ts_values = np.array(ts).flatten()

    # Prepare start date from month_year
    month_map = {
        "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
        "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
        "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12
    }
    month_num = month_map[month_year[0].upper()]
    start_date = pd.Timestamp(year=int(month_year[1]), month=month_num, day=1)

    # Build DataFrame with ds and y
    df = pd.DataFrame({
        "unique_id": ["ts"] * len(ts_values),
        "ds": pd.date_range(start=start_date, periods=len(ts_values), freq="W"),
        "y": ts_values
    })

    # Define models
    models = [Theta(), AutoETS(), CrostonClassic(), CrostonOptimized()]

    # Fit & predict
    sf = StatsForecast(models=models, freq="W", n_jobs=1)
    sf.fit(df)
    fcst = sf.predict(h=h)

    return {
        "Theta": fcst['Theta'].values,
        "AutoETS": fcst['AutoETS'].values,
        "CrostonClassic": fcst['CrostonClassic'].values,
        "CrostonOptimized": fcst['CrostonOptimized'].values
    }


def generar_serie_tiempo(deps,month_year_list):
    results = {}  # nested dictionary
    for dep in deps:
        results[dep] = {}
        for month_year in month_year_list:
            results[dep][month_year[0]] = {}
            for fc in funciones_cluster:
                funcion_name = fc.__name__
                ts_neighbors = fc(month_year[0], dep, 4, 2)
                forecasts_each_ts = []
                for ts in ts_neighbors:
                    ts_values = ts.values().flatten() 
                    f = forecast_1_month(ts_values,month_year)
                    print(f"mes: {month_year[0]}, año:{month_year[1]}, departamento:{dep}:\n\t{f}")
                    forecasts_each_ts.append(f)

                results[dep][month_year[0]][funcion_name] = forecasts_each_ts
            
generar_serie_tiempo(department_test,month_year_list_test)