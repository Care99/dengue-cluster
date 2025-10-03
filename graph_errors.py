import pandas as pd
import matplotlib as mplt; mplt.use('SVG',force=True)
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import root_mean_squared_error
plt.figure(figsize=(8,6))
headers=[
    'NOMBRE',
    'ALTO PARARANA',
    'AMAMBAY',
    'ASUNCION',
    'CAAGUAZU',
    'CENTRAL',
    'Centro est',
    'Centro norte',
    'Centro sur',
    'Chaco',
    'CORDILLERA',
    'Metropolitano',
    'PARAGUARI',
    'Paraguay',
    'PTE HAYES',
    'SAN PEDRO',
    'CANINDEYU',
    'CONCEPCION',
    'ITAPUA',
    'MISIONES',
    'BOQUERON',
    'GUAIRA',
    'CAAZAPA',
    'NEEMBUCU',
    'ALTO PARAGUAY'
    ]
cluster_clusters_cpto_error     =['CC_CPTO']
cluster_clusters_sarima_error   =['CC_SARIMA']
historical_cpto_error      =['H_CPTO']
historical_sarima_error    =['H_SARIMA']
knn_cpto_error             =['KNN_CPTO']
knn_sarima_error           =['KNN_SARIMA']
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
with open('errores_rmse.csv', 'w', newline='') as csvFile:
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(headers)
    csvWriter.writerow(cluster_clusters_cpto_error)
    csvWriter.writerow(cluster_clusters_sarima_error)
    csvWriter.writerow(historical_cpto_error)
    csvWriter.writerow(historical_sarima_error)
    csvWriter.writerow(knn_cpto_error)
    csvWriter.writerow(knn_sarima_error)
errors = pd.read_csv('errores_rmse.csv')
x = errors.columns[1:]
y = errors.iloc[:,1:].values
predictions = errors.iloc[0:,0].values

sorted_indices = np.argsort(y[0])[::-1]
x = x[sorted_indices]
y = y[:,sorted_indices]

colors = ['g','r','c','m','y','k']
plt.plot(x,y[0],'g',label=predictions[0])
plt.legend()
plt.plot(x,y[1],'r',label=predictions[1])
plt.legend()
plt.plot(x,y[2],'c--',label=predictions[2])
plt.legend()
plt.plot(x,y[3],'m--',label=predictions[3])
plt.legend()
plt.plot(x,y[4],'y--',label=predictions[4])
plt.legend()
plt.plot(x,y[5],'k--',label=predictions[5])
plt.legend()
    

plt.xlabel('Region')
plt.ylabel('Error calculated with RMSE')
plt.xticks(rotation=90, fontsize=5)
plt.title(f'Error in each region')
plt.grid(True)

# Save plot to SVG
plt.savefig('errores_rmse.svg', format="svg")
plt.close()
print(f'Plot saved as errores.svg')