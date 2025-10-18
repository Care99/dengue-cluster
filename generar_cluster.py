import pandas as pd
import numpy as np
import os

# Ventana de meses de octubre a septiembre
meses = ['OCTUBRE','NOVIEMBRE','DICIEMBRE','ENERO','FEBRERO',
            'MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE']

years = [2018,2019,2020,2021,2022]

departments = ['ALTO_PARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'CENTRO_EST','CENTRO_NORTE','CENTRO_SUR','CHACO','CORDILLERA',
              'METROPOLITANO','PARAGUARI','PARAGUAY','PTE_HAYES','SAN_PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO_PARAGUAY']

output_base = "csv/ts_historico"

def iniciar_ts():
    ts = []
    for dep in departments:
        ts[dep]= []

def generar_cluster_ventana():
    os.makedirs(output_base, exist_ok=True)
    # Hacer un loop para cada ventana
    for v in range(len(meses)-2):
        for y in years:
            ts= iniciar_ts()
            cols = iniciar_ts()
            for i in range(v,v+2):
                year = y if v < 3 else y + 1
                for depa in departments:
                    path = f'{output_base}/{year}/{meses[i]}/{depa}.csv'
                    data = pd.read_csv(path, header=None).apply(pd.to_numeric, errors='coerce')
                    ts[depa].extend(data.fillna(data.mean() if not data.mean() is None else 0).tolist())
                    cols[depa].extend([f"{meses[i]}_{j+1}" for j in range(len(data))])
        window_path = f'csv/matriz_ventana/{meses[v]}-{meses[v+1]}-{meses[v+2]}'
        for depa in departments:
            pd_depa = pd.DataFrame([ts[depa]], columns=cols[depa])
            pd_depa.to_csv(f"{window_path}/{depa}.csv", index=False)

generar_cluster_ventana()
