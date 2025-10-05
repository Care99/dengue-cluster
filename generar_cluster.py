import pandas as pd
import numpy as np
import os

# Ventana de meses de octubre a septiembre
ventana = ["Octubre","Noviembre","Diciembre","Enero","Febrero","Marzo",
           "Abril","Mayo","Junio","Julio","Agosto","Septiembre"]

def generar_cdc_departamento_md(csv_path='csv/dengue_ts.csv', output_base='csv/matrix_diff'):
    os.makedirs(output_base, exist_ok=True)

    # Leer CSV con series temporales por departamento y año
    data = pd.read_csv(csv_path)
    ts_columns = data.columns[2:]
    
    # Convertir columnas de series a numérico
    data[ts_columns] = data[ts_columns].apply(pd.to_numeric, errors='coerce')
    
    # Lista de departamentos
    departments = data['Department'].unique()
    
    for dept in departments:
        dept_data = data[data['Department'] == dept].copy()
        dept_data = dept_data.iloc[:4]  # solo primeros 4 años
        years = dept_data['Pandemic_year'].tolist()
        series_matrix = dept_data[ts_columns].to_numpy()
        n_years = series_matrix.shape[0] - 1
        
        # Crear carpetas por mes
        for month in ventana:
            month_folder = os.path.join(output_base, month)
            os.makedirs(month_folder, exist_ok=True)
        
        # Ciclo por cada mes de la ventana
        for month_idx, month in enumerate(ventana):
            diff_matrix = np.zeros((n_years, n_years))
            
            for i in range(n_years):
                ts_actual = series_matrix[i]
                ts_prev = series_matrix[i-1]
                
                # Construir ventana móvil de 3 meses para el año actual
                if month_idx == 0:  # Octubre
                    # Tomar últimos 2 meses del año anterior + Octubre del año actual
                    if ts_prev is not None:
                        window_i = [ts_prev[-2], ts_prev[-1], ts_actual[month_idx]]
                    else:  # si no hay año anterior, rellenar con 0
                        window_i = [0, 0, ts_actual[month_idx]]
                elif month_idx == 1:  # Noviembre
                    # Tomar último mes del año anterior + Octubre y Noviembre del año actual
                    if ts_prev is not None:
                        window_i = [ts_prev[-1], ts_actual[month_idx-1], ts_actual[month_idx]]
                    else:
                        window_i = [0, ts_actual[month_idx-1], ts_actual[month_idx]]
                else:
                    # Para los demás meses, tomar los 3 meses consecutivos del mismo año
                    window_i = ts_actual[month_idx-2:month_idx+1]
                
                # Comparar la ventana del año actual con los otros años
                for j in range(n_years):
                    ts_j = series_matrix[j]
                    ts_j_prev = series_matrix[j-1] if j > 0 else None
                    
                    # Construir la ventana móvil de 3 meses para el año j
                    if month_idx == 0:
                        if ts_j_prev is not None:
                            window_j = [ts_j_prev[-2], ts_j_prev[-1], ts_j[month_idx]]
                        else:
                            window_j = [0, 0, ts_j[month_idx]]
                    elif month_idx == 1:
                        if ts_j_prev is not None:
                            window_j = [ts_j_prev[-1], ts_j[month_idx-1], ts_j[month_idx]]
                        else:
                            window_j = [0, ts_j[month_idx-1], ts_j[month_idx]]
                    else:
                        window_j = ts_j[month_idx-2:month_idx+1]
                    
                    # Normalizar las ventanas (Bhattacharyya requiere distribución de probabilidad)
                    w_i_norm = np.array(window_i) / np.sum(window_i) if np.sum(window_i) > 0 else np.zeros(3)
                    w_j_norm = np.array(window_j) / np.sum(window_j) if np.sum(window_j) > 0 else np.zeros(3)
                    
                    # Calcular distancia Bhattacharyya
                    bc = np.sum(np.sqrt(w_i_norm * w_j_norm))
                    bc = max(bc, 1e-10)  # evitar log(0)
                    diff_matrix[i, j] = -np.log(bc)
            
            # Convertir a DataFrame y guardar
            diff_df = pd.DataFrame(diff_matrix, index=years, columns=years)
            output_file = os.path.join(output_base, month, f"{dept.replace(' ','_')}_md_{month}.csv")
            diff_df.to_csv(output_file)
            print(f"Guardado matriz: {output_file}")

# Ejecutar
generar_cdc_departamento_md()
