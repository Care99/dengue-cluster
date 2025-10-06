import pandas as pd
import matplotlib as mplt; mplt.use('SVG',force=True)
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import root_mean_squared_error
plt.figure(figsize=(8,6))
headers=[
    'NOMBRE',
    'ALTO_PARANA',
    'AMAMBAY',
    'ASUNCION',
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
cluster_clusters_cpto_error     =[]
cluster_clusters_sarima_error   =[]
historical_cpto_error      =[]
historical_sarima_error    =[]
knn_cpto_error             =[]
knn_sarima_error           =[]
for i in range(1,25):
    file_index = str(i)
    if(i<10):
        file_index='0'+str(i)
    data                            =pd.read_csv(f'2022_{file_index}.csv',header=None)
    cluster_clusters_cpto_data      =pd.read_csv(f'cluster_clusters_time_series_cpto{file_index}.csv',header=None)
    cluster_clusters_sarima_data    =pd.read_csv(f'cluster_clusters_time_series_sarima{file_index}.csv',header=None)
    historical_cpto_data            =pd.read_csv(f'historical_time_series_cpto{file_index}.csv',header=None)
    historical_sarima_data          =pd.read_csv(f'historical_time_series_sarima{file_index}.csv',header=None)
    knn_cpto_data                   =pd.read_csv(f'knn_time_series_cpto{file_index}.csv',header=None)
    knn_sarima_data                 =pd.read_csv(f'knn_time_series_sarima{file_index}.csv',header=None)
    
    cluster_clusters_cpto_error.append(root_mean_squared_error(data,cluster_clusters_cpto_data))
    cluster_clusters_sarima_error.append(root_mean_squared_error(data,cluster_clusters_sarima_data))
    historical_cpto_error.append(root_mean_squared_error(data,historical_cpto_data))
    historical_sarima_error.append(root_mean_squared_error(data,historical_sarima_data))
    knn_cpto_error.append(root_mean_squared_error(data,knn_cpto_data))
    knn_sarima_error.append(root_mean_squared_error(data,knn_sarima_data))
promedio_cc_cpto = sum(cluster_clusters_cpto_error)/24
promedio_cc_sarima  = sum(cluster_clusters_sarima_error)/24
promedio_historico_cpto = sum(historical_cpto_error)/24
promedio_historico_sarima  = sum(historical_sarima_error)/24
promedio_knn_cpto = sum(knn_cpto_error)/24
promedio_knn_sarima  = sum(knn_sarima_error)/24
print(f'El promedio para Clusters de Clusters aplicando CPTO es{promedio_cc_cpto}')
print(f'El promedio para Clusters de Clusters aplicando SARIMA es{promedio_cc_sarima}')
print(f'El promedio para historico aplicando CPTO es{promedio_historico_cpto}')
print(f'El promedio para historico de Clusters aplicando SARIMA es{promedio_historico_sarima}')
print(f'El promedio para knn aplicando CPTO es{promedio_knn_cpto}')
print(f'El promedio para knn aplicando SARIMA es{promedio_knn_sarima}')