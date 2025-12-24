import pandas as pd
import numpy as np
import os
import math
from darts import TimeSeries
import fastdtw
from datetime import datetime, timedelta
from src.utils.constants import departments, csv_path
from src.utils.time_series import get_ts
# Ventana de meses de octubre a septiembre
meses = ['JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE','ENERO','FEBRERO',
            'MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO']

years = [2019,2020,2021,2022]
start_date_index =["2019-07-13","2020-07-04","2021-07-10","2022-07-09"]
years_folders = ["2019-2020","2020-2021","2021-2022","2022-2023"]

input_base = "csv/ts_historico"

excel_name = 'casos.csv'
excel_file = os.path.join(csv_path,excel_name)
unformatted_data = pd.read_csv(excel_file)
data = unformatted_data.copy()
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

def iniciar_ts():
    ts = {}
    for d in range(len(departments)):
        ts[d]= []
    return ts


def generar_cluster_matriz_diferencia()->None:
    for week_index in range(0,53):
        #print(f"\n Semana: {week_index}")
        ts_dict = {}   # cumulative dictionary
        for year_folder in years_folders:
            # Generate full paths and read CSVs in one line
            row = {f"{d}_{year_folder}": get_ts(year=year_folder, week=f'{week_index}', department=d) for d in departments}
            ts_dict.update(row)
        names = list(ts_dict.keys())
        #print(names)
        #print("\n")
        n = len(names)
        #print("len is " + str(n))
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0  # diagonal = 0
                else:
                    matrix[i, j],path = fastdtw.fastdtw(ts_dict[names[i]], ts_dict[names[j]])

        # Save as CSV with headers
        output_path=f"csv/c/cluster_matriz/week_{week_index}"
        os.makedirs(output_path,exist_ok=True)

        df = pd.DataFrame(matrix, index=names, columns=names)
        file_nime = "mat_distancia.csv"
        output_path = os.path.join(output_path,file_nime)
        df.to_csv(output_path)
        #print(f"guardado: {output_path}")


def get_cluster(semana:str, departamento:str, k:int, n:int)->list[list[float]]:
    label = departamento + "_2022-2023"
    knn = k*n

    file_path = f'csv/c/cluster_matriz/week_{semana}/mat_distancia.csv'
    df = pd.read_csv(file_path, sep=",", index_col=0)
    
    distances = df.loc[label].copy()
    nearest_idx = distances.nsmallest(knn).index.tolist()
    #print(nearest_idx)
   
    knn_ts:list[list[float]] = []
    for dept in nearest_idx:
        year = int(dept.split("-")[1])-1
        department = dept.split("-")[0][:-5]
        #print(f"Fetching TS for Year: {year}, Department: {department}")
        ts=get_ts(year=f'{year}-{year+1}', week=semana, department=department)
        knn_ts.append(ts)
    #print(knn_ts)
    return knn_ts

#generar_cluster_ventana()
#generar_cluster_matriz_diferencia()
#get_cluster(semana="0",departamento="CENTRO_SUR", k=2, n=4)
