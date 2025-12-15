import pandas as pd
import numpy as np
import os
import math
from darts import TimeSeries
from fastdtw import fastdtw as dtw
from datetime import datetime, timedelta
from utils.constants import departments
# Ventana de meses de octubre a septiembre
meses = ['JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE','ENERO','FEBRERO',
            'MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO']

years = [2019,2020,2021,2022]
start_date_index =["2019-07-13","2020-07-04","2021-07-10","2022-07-09"]
years_folders = ["2019-2020","2020-2021","2021-2022","2022-2023"]

input_base = "csv/ts_historico"

matriz_ventana = [
    "ABRIL-MAYO-JUNIO",
    "DICIEMBRE-ENERO-FEBRERO",
    "ENERO-FEBRERO-MARZO",
    "FEBRERO-MARZO-ABRIL",
    "JULIO-AGOSTO-SEPTIEMBRE",
    "JUNIO-JULIO-AGOSTO",
    "MARZO-ABRIL-MAYO",
    "MAYO-JUNIO-JULIO",
    "NOVIEMBRE-DICIEMBRE-ENERO",
    "OCTUBRE-NOVIEMBRE-DICIEMBRE",
    "AGOSTO-SEPTIEMBRE-OCTUBRE",
    "SEPTIEMBRE-OCTUBRE-NOVIEMBRE"
]

dict_ventana = {
    "JULIO": "ABRIL-MAYO-JUNIO",
    "MARZO": "DICIEMBRE-ENERO-FEBRERO",
    "ABRIL": "ENERO-FEBRERO-MARZO",
    "MAYO": "FEBRERO-MARZO-ABRIL",
    "OCTUBRE": "JULIO-AGOSTO-SEPTIEMBRE",
    "SEPTIEMBRE": "JUNIO-JULIO-AGOSTO",
    "JUNIO": "MARZO-ABRIL-MAYO",
    "AGOSTO": "MAYO-JUNIO-JULIO",
    "FEBRERO": "NOVIEMBRE-DICIEMBRE-ENERO",
    "ENERO": "OCTUBRE-NOVIEMBRE-DICIEMBRE",
    "NOVIEMBRE": "AGOSTO-SEPTIEMBRE-OCTUBRE",
    "DICIEMBRE": "SEPTIEMBRE-OCTUBRE-NOVIEMBRE"
}
script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
excel_name = 'casos.csv'
excel_file = os.path.join(csv_path,excel_name)
unformatted_data = pd.read_csv(excel_file)
data = unformatted_data.copy()
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
def bhattacharyya(tseries1,tseries2):
    value = 0.0
    i = 0
    min_len = min(len(tseries1),len(tseries2))
    for i in range(min_len):
        value += math.sqrt(tseries1[i]*tseries2[i])
        if value == 0:
            value += 1e-12
    value = - np.log(value)
    return abs(value)

def iniciar_ts():
    ts = {}
    for d in range(len(departments)):
        ts[d]= []
    return ts
        

def generar_cluster_ventana():
    os.makedirs(input_base, exist_ok=True)
    # Hacer un loop para cada ventana
    for v in range(len(meses)-2):
        for y in years:
            ts= iniciar_ts()
            cols = iniciar_ts()
            for i in range(v,v+3):
                year = y if i < 5 else y + 1
                for d in range(len(departments)):
                    path = f'{input_base}/{year}/{meses[i]}/{departments[d]}.csv'
                    #print(f"procesando {path}")
                    data = pd.read_csv(path, header=None).apply(pd.to_numeric, errors='coerce')
                    fila = data.iloc[0].tolist()
                    #print(f"fila es {fila}")
                    ts[d].extend(data.iloc[0].tolist())
                    cols[d].extend([f"{meses[i]}_{j+1}" for j in range(len(data.iloc[0]))])
            window_path = f'csv/c/matriz_ventana/{meses[v]}-{meses[v+1]}-{meses[v+2]}/{y}-{y+1}'
            os.makedirs(window_path,exist_ok=True)
            for d in range(len(departments)):
                pd_depa = pd.DataFrame([ts[d]], columns=cols[d])
                pd_depa.to_csv(os.path.join(window_path,f'{departments[d]}.csv'), index=False)
                #print(f'saved: {window_path}/{departments[d]}.csv')


def generar_cluster_matriz_diferencia():
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
                    matrix[i, j],path = dtw(ts_dict[names[i]], ts_dict[names[j]])

        # Save as CSV with headers
        output_path=f"csv/c/cluster_matriz/week_{week_index}"
        os.makedirs(output_path,exist_ok=True)

        df = pd.DataFrame(matrix, index=names, columns=names)
        file_nime = "mat_distancia.csv"
        output_path = os.path.join(output_path,file_nime)
        df.to_csv(output_path)
        #print(f"guardado: {output_path}")


def get_cluster(semana:str, departamento:str, k:int, n:int):
    label = departamento + "_2022-2023"
    knn = k*n

    file_path = f'csv/c/cluster_matriz/week_{semana}/mat_distancia.csv'
    df = pd.read_csv(file_path, sep=",", index_col=0)
    
    distances = df.loc[label].copy()
    nearest_idx = distances.nsmallest(knn).index.tolist()
    #print(nearest_idx)
   
    knn_ts = []
    for dept in nearest_idx:
        year = int(dept.split("-")[1])-1
        department = dept.split("-")[0][:-5]
        #print(f"Fetching TS for Year: {year}, Department: {department}")
        ts=TimeSeries.from_values(get_ts(year=f'{year}-{year+1}', week=semana, department=department))
        knn_ts.append(ts)
    #print(knn_ts)
    return knn_ts

def get_ts(year:str, week:str, department:str):
    ts_dict = {}
    start_date = datetime.strptime(start_date_index[int(year.split('-')[0])-2019],'%Y-%m-%d') + timedelta(weeks=int(week))
    end_date = start_date + timedelta(weeks=11)
    filtered_data = data[
        (data['disease'] == "DENGUE") 
        & (data['classification'] == "TOTAL") 
        & (data['name'] == department)]
    filtered_data = filtered_data.copy()
    filtered_data['date'] = pd.to_datetime(filtered_data['date'], format='%Y-%m-%d')
    range_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='both')]
    ts_dict[department] = range_data.reset_index(drop=True)
    return ts_dict[department]['incidence'].values

#generar_cluster_ventana()
#generar_cluster_matriz_diferencia()
#get_cluster(semana="0",departamento="CENTRO_SUR", k=2, n=4)
