import pandas as pd
import numpy as np
import os
import math
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib as mplt
import matplotlib.pyplot as plt
mplt.use('SVG',force=True)
from scipy.spatial.distance import pdist, squareform
from darts import TimeSeries
from datetime import datetime, timedelta
from src.utils.constants import departments
from src.utils.time_series import get_ts
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
def generar_cluster_ventana():
    os.makedirs(input_base, exist_ok=True)
    # Hacer un loop para cada ventana
    for week_index in range(0,53):
        row_dict = {}  # dictionary for one row per window
        for year_index in years:
            for department in departments:
                #print(f"procesando {year_index}-{year_index+1} semana {week_index} departamento {department}")
                data = get_ts(f'{year_index}-{year_index+1}', str(week_index), department)
                col_name=f"{department}_{year_index}-{year_index+1}"
                row_dict[col_name] = np.mean(data)
        # Convert the row dictionary to a single-row DataFrame
        df = pd.DataFrame([row_dict])
        # Save the CSV
        window_path = f'csv/cj/window_values'
        os.makedirs(window_path, exist_ok=True)
        file_name = f"week_{week_index}.csv"
        df.to_csv(os.path.join(window_path,file_name ), index=False)
        #print(f"Saved: {window_path}/{file_name}")


def generar_cluster_jerarquico():
    for week_index in range(0,53):
        hclust_dir = f"csv/cj/hclust/week_{week_index}"
        os.makedirs(hclust_dir, exist_ok=True)

        # Load CSV
        df = pd.read_csv(f"csv/cj/window_values/week_{week_index}.csv")

        # Transpose: index = department_year, column=value
        df_t = df.T
        df_t.columns = ["value"]

        # Hierarchical clustering
        Z = linkage(df_t, method='ward', metric='euclidean')

        # Save dendrogram SVG
        out_path = "csv/cj/figures"
        os.makedirs(out_path, exist_ok=True)
        plt.figure(figsize=(20, 8))
        dendrogram(Z, labels=df_t.index.tolist(), leaf_rotation=90)
        plt.title(f"Hierarchical Clustering: {week_index}")
        plt.tight_layout()
        plt.savefig(f"{out_path}/week_{week_index}.svg", format="svg")
        plt.close()
        #print(f"Saved dendrogram to {out_path}/week_{week_index}.svg")

        # ---- SAVE LINKAGE MATRIX CORRECTLY ----
        np.save(f"{hclust_dir}/Z.npy", Z)
        #print(f"Saved linkage matrix to {hclust_dir}/Z.npy")

        # ---- SAVE LABEL ORDER CORRECTLY ----
        df_t.to_csv(f"{hclust_dir}/labels.csv")
        #print(f"Saved labels to {hclust_dir}/labels.csv")

def get_cluster_jerarquico(semana:str, departamento:str, k:int, n:int):
    hclust_dir = "csv/cj/hclust"
    label = departamento + "_2022-2023"
    knn = k*n
    # Load saved clustering state
    Z = np.load(f"{hclust_dir}/week_{semana}/Z.npy")
    df_t = pd.read_csv(f"{hclust_dir}/week_{semana}/labels.csv", index_col=0)
    # Compute pairwise distances using the SAME metric
    dist_matrix = squareform(pdist(df_t[["value"]], metric="euclidean"))
    labels = df_t.index.tolist()
    
    idx = labels.index(label)
    distances = dist_matrix[idx]

    nearest_idx = distances.argsort()[0:knn]
    knn_ts = []
    
    for i in range(len(nearest_idx)):
        label_i = labels[nearest_idx[i]]
        year = int(label_i.split("-")[1])-1
        department = label_i.split("-")[0][:-5]
        #print(f"Fetching TS for Year: {year}, Department: {department}")
        ts=TimeSeries.from_values(get_ts(year=f'{year}-{year+1}',week=semana, department=department))
        knn_ts.append(ts)
    #print(knn_ts)
    return knn_ts


#generar_cluster_ventana()
#generar_cluster_jerarquico()
#get_cluster_jerarquico(semana='0',departamento="CENTRAL", k=2, n=4)
