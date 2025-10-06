import os
import pandas as pd
import matplotlib as mplt; mplt.use('SVG',force=True)
import matplotlib.pyplot as plt
departments = ['ALTO PARARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'CENTRO_EST','CENTRO_NORTE','CENTRO_SUR','CHACO','CORDILLERA',
              'METROPOLITANO','PARAGUARI','PARAGUAY','PTE_HAYES','SAN_PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO_PARAGUAY']
time_series_2022 = '2022'
cluster_clusters = '_cpto'
historical = 'historical_time_series'
knn = 'historical_time_series'
for i in range(1,25):
    region = str(i)
    if(i<=9):
        region = '0'+str(i)
    plt.figure()
    x = list(range(1,53))
    time_series1 = pd.read_csv(f'2022_{region}.csv', header=None)
    y = time_series1.iloc[0,:].values
    plt.plot(x,y,'b',label='Datos reales')
    plt.legend()
    time_series2 = pd.read_csv(f'cluster_clusters_time_series_cpto{region}.csv', header=None)
    y = time_series2.iloc[0,:].values
    plt.plot(x,y,'g',label='CC_CPTO')
    plt.legend()
    time_series3 = pd.read_csv(f'cluster_clusters_time_series_sarima{region}.csv', header=None)
    y = time_series3.iloc[0,:].values
    plt.plot(x,y,'r',label='CC_SARIMA')
    plt.legend()
    time_series4 = pd.read_csv(f'historical_time_series_cpto{region}.csv', header=None)
    y = time_series4.iloc[0,:].values
    plt.plot(x,y,'c--',label='H_CPTO')
    plt.legend()
    time_series5 = pd.read_csv(f'historical_time_series_sarima{region}.csv', header=None)
    y = time_series5.iloc[0,:].values
    plt.plot(x,y,'m--',label='H_SARIMA')
    plt.legend()
    time_series6 = pd.read_csv(f'knn_time_series_cpto{region}.csv', header=None)
    y = time_series6.iloc[0,:].values
    plt.plot(x,y,'y--',label='KNN_CPTO')
    plt.legend()
    time_series7 = pd.read_csv(f'knn_time_series_sarima{region}.csv', header=None)
    y = time_series7.iloc[0,:].values
    plt.plot(x,y,'k--',label='KNN_SARIMA')
    plt.legend()
    plt.xlabel("Semanas")
    plt.ylabel("Incidencia")
    plt.title(f'{departments[i-1]}')
    plt.grid(True)

    # Save plot to SVG
    plt.savefig(f'{departments[i-1]}.svg', format="svg")
    plt.close()
    print(f'Plot saved as {departments[i-1]}.svg')