import pandas as pd
import numpy as np
import os
import math

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

def bhattacharyya(tseries1,tseries2):
    value = 0.0
    i = 0
    for i in range(len(tseries1)):
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
                    print(f"procesando {path}")
                    data = pd.read_csv(path, header=None).apply(pd.to_numeric, errors='coerce')
                    fila = data.iloc[0].tolist()
                    print(f"fila es {fila}")
                    ts[d].extend(data.iloc[0].tolist())
                    cols[d].extend([f"{meses[i]}_{j+1}" for j in range(len(data.iloc[0]))])
            window_path = f'csv/cdc/matriz_ventana/{meses[v]}-{meses[v+1]}-{meses[v+2]}/{y}-{y+1}'
            os.makedirs(window_path,exist_ok=True)
            for d in range(len(departments)):
                pd_depa = pd.DataFrame([ts[d]], columns=cols[d])
                pd_depa.to_csv(os.path.join(window_path,f'{departments[d]}.csv'), index=False)
                print(f'saved: {window_path}/{departments[d]}.csv')

def generar_cluster_matriz_diferencia():
    for m in matriz_ventana:
        for year_folder in years_folders:
            folder_path = f"csv/cdc/matriz_ventana/{m}/{year_folder}"
            # Generate full paths and read CSVs in one line
            ts_dict = {d: pd.read_csv(os.path.join(folder_path, d + ".csv")).iloc[0].values for d in departments}

            # Compute Bhattacharyya distance matrix
            names = list(ts_dict.keys())
            n = len(names)
            matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i, j] = 0  # diagonal = 0
                    else:
                        matrix[i, j] = bhattacharyya(ts_dict[names[i]], ts_dict[names[j]])

            # Save as CSV with headers
            output_path=f"csv/cdc/cluster_matriz/{m}"
            os.makedirs(output_path,exist_ok=True)

            df = pd.DataFrame(matrix, index=names, columns=names)
            df.to_csv(os.path.join(output_path, f"{year_folder}.csv"))
            print(f"saved: {output_path}/{year_folder}")


def generar_cluster_de_cluster_matriz_diferencia():
        for m in matriz_ventana:
            folder_path = f"csv/cdc/cluster_matriz/{m}"
            ts_dict = {}
            for y in years_folders:
                df = pd.read_csv(os.path.join(folder_path, y + ".csv"), index_col=0)
                ts_dict[y] = df.iloc[0].astype(float).values  # convert to float

            # Compute Bhattacharyya distance matrix
            names = list(ts_dict.keys())
            n = len(names)
            matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i, j] = 0
                    else:
                        matrix[i, j] = bhattacharyya(ts_dict[names[i]], ts_dict[names[j]])

            # Save as CSV with headers
            df = pd.DataFrame(matrix, index=names, columns=names)
            df.to_csv(os.path.join(folder_path, "cluster_de_cluster.csv"))
            print(f"saved: {folder_path}/cluster_de_cluster.csv")


generar_cluster_ventana()
generar_cluster_matriz_diferencia()
generar_cluster_de_cluster_matriz_diferencia()
