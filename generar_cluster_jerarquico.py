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

# Ventana de meses de octubre a septiembre
meses = ['JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE','ENERO','FEBRERO',
            'MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO']

years = [2019,2020,2021,2022]

years_folders = ["2019-2020","2020-2021","2021-2022","2022-2023"]

departments = ['ALTO_PARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'CENTRO_EST','CENTRO_NORTE','CENTRO_SUR','CHACO','CORDILLERA',
              'METROPOLITANO','PARAGUARI','PARAGUAY','PTE_HAYES','SAN_PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO_PARAGUAY']

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

def generar_cluster_ventana():
    os.makedirs(input_base, exist_ok=True)
    # Hacer un loop para cada ventana
    for v in range(len(meses)-2):
        row_dict = {}  # dictionary for one row per window
        for y in years:
            for d in range(len(departments)):
                mes_value=[]
                for i in range(v,v+3):
                    year = y if i < 5 else y + 1
                    path = f'{input_base}/{year}/{meses[i]}/{departments[d]}.csv'
                    print(f"procesando {path}")
                    data = pd.read_csv(path, header=None).apply(pd.to_numeric, errors='coerce')
                    mes_value.append(data.iloc[0].mean())  # <-- fix here
                col_name=f"{departments[d]}_{y}-{y+1}"
                row_dict[col_name] = np.mean(mes_value)
        # Convert the row dictionary to a single-row DataFrame
        df = pd.DataFrame([row_dict])
        # Save the CSV
        window_path = f'csv/cj/window_values'
        os.makedirs(window_path, exist_ok=True)
        file_name = f"{meses[v]}-{meses[v+1]}-{meses[v+2]}.csv"
        df.to_csv(os.path.join(window_path,file_name ), index=False)
        print(f"Saved: {window_path}/{file_name}")


def generar_cluster_jerarquico():
    for m in matriz_ventana:
        hclust_dir = f"csv/cj/hclust/{m}"
        os.makedirs(hclust_dir, exist_ok=True)

        # Load CSV
        df = pd.read_csv(f"csv/cj/window_values/{m}.csv")

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
        plt.title(f"Hierarchical Clustering: {m}")
        plt.tight_layout()
        plt.savefig(f"{out_path}/{m}.svg", format="svg")
        plt.close()
        print(f"Saved dendrogram to {out_path}/{m}.svg")

        # ---- SAVE LINKAGE MATRIX CORRECTLY ----
        np.save(f"{hclust_dir}/Z.npy", Z)
        print(f"Saved linkage matrix to {hclust_dir}/{m}_Z.npy")

        # ---- SAVE LABEL ORDER CORRECTLY ----
        df_t.to_csv(f"{hclust_dir}/labels.csv")
        print(f"Saved labels to {hclust_dir}/{m}_labels.csv")

def get_cluster_jerarquico(mes:str, departamento:str, k:int, n:int):
    hclust_dir = "csv/cj/hclust"
    meses_str = dict_ventana[mes]
    label = departamento + "_2022-2023"
    knn = k*n
    # Load saved clustering state
    Z = np.load(f"{hclust_dir}/{meses_str}/Z.npy")
    df_t = pd.read_csv(f"{hclust_dir}/{meses_str}/labels.csv", index_col=0)
    # Compute pairwise distances using the SAME metric
    dist_matrix = squareform(pdist(df_t[["value"]], metric="euclidean"))
    labels = df_t.index.tolist()
    
    idx = labels.index(label)
    distances = dist_matrix[idx]

    nearest_idx = distances.argsort()[0:knn]
    knn_ts = []
    
    for i in range(len(nearest_idx)):
        label_i = labels[nearest_idx[i]]  
        ts=TimeSeries.from_values(get_ts(meses_str, label_i))
        knn_ts.append(ts)
    #print(knn_ts)
    return knn_ts


def get_ts(meses_str: str, department_year: str) :
    months = meses_str.split("-")
    BASE_PATH = "csv/ts_historico"
    ts_data = []
    department, years_str = department_year.rsplit('_', 1)
    for mes in months:
        pos = 1 if meses.index(mes) > 5 else 0
        year = years_str.split("-")[pos]
        csv_path = os.path.join(BASE_PATH, str(year), mes, f"{department}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} does not exist")    
        # CSV has no header, assume single column
        df = pd.read_csv(csv_path, header=None)
        ts_data.extend(pd.Series(df.values.flatten()))
    return ts_data[:12]


#generar_cluster_ventana()
#generar_cluster_jerarquico()
#get_k_n_n(mes="ABRIL",departamento="CENTRAL", k=2, n=4)