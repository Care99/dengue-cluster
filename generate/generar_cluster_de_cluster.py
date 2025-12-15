import pandas as pd
import numpy as np
import os
import math
from darts import TimeSeries
import fastdtw
from scipy.spatial.distance import euclidean
from datetime import datetime, timedelta
from utils.constants import departments
# Ventana de meses de octubre a septiembre
meses = ['JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE','ENERO','FEBRERO',
            'MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO']

years = [2019,2020,2021,2022]
start_date_index =["2019-07-13","2020-07-04","2021-07-10","2022-07-09"]
years_folders = ["2019-2020","2020-2021","2021-2022","2022-2023"]

input_base = "csv/ts_historico"

script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
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
        
def generar_cluster_matriz_diferencia():
    for week_index in range(0,53):
        for year_index in range(0,4):
            #Calculate how many seconds a week has
            start_date = datetime.strptime(start_date_index[year_index],'%Y-%m-%d') + timedelta(weeks=week_index)
            end_date = start_date + timedelta(weeks=11)
            #unformatted_data = pd.read_csv(excel_file)
            #data = unformatted_data.copy()
            #data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
            #Load array of time series for each department from pd.DataFrame data
            #Filter 'disease' == "DENGUE" and 'classification' == "TOTAL"
            #Filter 'date' between start_date and end_date
            #Return a dictionary of time series for each 'name'
            ts_dict = {}
            for department in departments:
                filtered_data = data[
                    (data['disease'] == "DENGUE") 
                    & (data['classification'] == "TOTAL") 
                    & (data['name'] == department)]
                filtered_data = filtered_data.copy()
                filtered_data['date'] = pd.to_datetime(filtered_data['date'], format='%Y-%m-%d')
                range_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='both')]
                ts_dict[department] = range_data.reset_index(drop=True)
            

            # Compute Bhattacharyya distance matrix
            names = list(ts_dict.keys())
            n = len(names)
            matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    # Extract time series from column 'incidences' of ts_dict
                    time_series_i = ts_dict[names[i]]['incidence'].values
                    time_series_j = ts_dict[names[j]]['incidence'].values
                    if i == j:
                        matrix[i, j] = 0  # diagonal = 0
                    else:
                        matrix[i, j],path = fastdtw.fastdtw(time_series_i, time_series_j)

            # Save as CSV with headers
            output_path=f"csv/cdc/cluster_matriz/week_{week_index}"
            output_filename=f"{years_folders[year_index]}.csv"
            output=os.path.join(output_path,output_filename)
            os.makedirs(output_path,exist_ok=True)

            df = pd.DataFrame(matrix, index=names, columns=names)
            df.to_csv(output)
            #print(f"saved: {output}")


def generar_cluster_de_cluster_matriz_diferencia():
        for week_index in range(0,53):
            folder_path = f"csv/cdc/cluster_matriz/week_{week_index}"
            ts_dict = {}
            for year_index in years_folders:
                df = pd.read_csv(os.path.join(folder_path, year_index + ".csv"), index_col=0)
                ts_dict[year_index] = df.iloc[0].astype(float).values  # convert to float

            # Compute Bhattacharyya distance matrix
            names = list(ts_dict.keys())
            n = len(names)
            matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i, j] = 0
                    else:
                        matrix[i, j],path = fastdtw.fastdtw(ts_dict[names[i]], ts_dict[names[j]])

            # Save as CSV with headers
            df = pd.DataFrame(matrix, index=names, columns=names)
            df.to_csv(os.path.join(folder_path, "cluster_de_cluster.csv"))
            #print(f"saved: {folder_path}/cluster_de_cluster.csv")


def get_cluster_de_clusters(semana:str, departamento:str, k:int, n:int):
    #obtener k anhos mas cercanos
    k_label = "2022-2023"
    k_file_path = f'csv/cdc/cluster_matriz/week_{semana}/cluster_de_cluster.csv'
    k_df = pd.read_csv(k_file_path, sep=",", index_col=0)
    k_distances = k_df.loc[k_label].copy()
    k_nearest_years = k_distances.nsmallest(k).index.tolist()

    #obtener n depts mas cercanos
    n_nearest_neighbours = []
    for k_year in k_nearest_years:
        n_label = departamento
        n_file_path = f'csv/cdc/cluster_matriz/week_{semana}/{k_year}.csv'
        n_df = pd.read_csv(n_file_path, sep=",", index_col=0)
        n_distances = n_df.loc[n_label].copy()
        n_nearest_neighbours.append(n_distances.nsmallest(n).index.tolist())

    #etiquetar los knn mas cercanos
    knn_labels = []
    for year, dep_list in zip(k_nearest_years, n_nearest_neighbours):
        for dep in dep_list:
            knn_labels.append([year, dep])
    #print(knn_labels)

    knn_ts = []
    for dept in knn_labels:
        #print(dept)
        ts=TimeSeries.from_values(get_ts(year=dept[0], week=semana, department=dept[1]))
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
#generar_cluster_de_cluster_matriz_diferencia()
#get_cluster_de_clusters(semana='0',departamento="CENTRO_SUR", k=2, n=4) #K = YEAR, N= locations
