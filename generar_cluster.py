import pandas as pd
import numpy as np
import os

# Ventana de meses de octubre a septiembre
ventana = [
    "Octubre","Noviembre","Diciembre","Enero","Febrero","Marzo",
    "Abril","Mayo","Junio","Julio","Agosto","Septiembre"
]

# Map each month to approximate last week index (for 3-week window)
meses_a_ult_semana = {
    "Octubre": 40, "Noviembre": 44, "Diciembre": 48,
    "Enero": 4, "Febrero": 8, "Marzo": 12,
    "Abril": 16, "Mayo": 20, "Junio": 24,
    "Julio": 28, "Agosto": 32, "Septiembre": 36
}

def generar_cdc_departamento_mes(csv_path='csv/dengue_ts_historico.csv',
                                 output_base='csv/matrix_diff'):
    os.makedirs(output_base, exist_ok=True)

    # Leer CSV histórico
    data = pd.read_csv(csv_path)

    # Columnas con series
    ts_columns = data.columns[1:]
    data[ts_columns] = data[ts_columns].apply(pd.to_numeric, errors='coerce')

    # Ordenar columnas cronológicamente
    ts_columns_sorted = sorted(ts_columns, key=lambda x: (int(x.split('_')[0]), int(x.split('_week_')[1])))
    data = data[['Department'] + ts_columns_sorted]

    # Lista de años disponibles
    years = sorted(list({int(c.split('_')[0]) for c in ts_columns_sorted}))

    # Procesar cada departamento
    for dept in data['Department'].unique():
        dept_series = data[data['Department'] == dept].iloc[0, 1:]
        dept_series.index = ts_columns_sorted

        # Extraer serie por año, excluyendo 2023
        yearly_series = {
            year: dept_series[[c for c in ts_columns_sorted if c.startswith(str(year))]].to_numpy(dtype=float)
            for year in years if year < 2022
        }


        for month, week_end in meses_a_ult_semana.items():
            month_folder = os.path.join(output_base, month)
            os.makedirs(month_folder, exist_ok=True)


            # Lista de años disponibles (sin 2023)
            available_years = list(yearly_series.keys())
            n_years = len(available_years)
            diff_matrix = np.zeros((n_years, n_years))


            for i in range(n_years):
                ts_i = yearly_series[available_years[i]]
                ts_i_prev = yearly_series[available_years[i-1]] if i > 0 else np.zeros_like(ts_i)

                start_prev = max(0, week_end - 8)
                window_i = np.array(np.concatenate([ts_i_prev[start_prev:], ts_i[:4]]), dtype=float)

                for j in range(n_years):
                    ts_j = yearly_series[available_years[j]]
                    ts_j_prev = yearly_series[available_years[j-1]] if j > 0 else np.zeros_like(ts_j)
                    window_j = np.array(np.concatenate([ts_j_prev[start_prev:], ts_j[:4]]), dtype=float)

                    if np.sum(window_i) == 0 or np.sum(window_j) == 0:
                        diff_matrix[i, j] = 0
                    else:
                        w_i_norm = window_i / np.sum(window_i)
                        w_j_norm = window_j / np.sum(window_j)
                        bc = np.sum(np.sqrt(w_i_norm * w_j_norm))
                        diff_matrix[i, j] = -np.log(max(bc, 1e-10))

            # Guardar CSV
            diff_df = pd.DataFrame(diff_matrix, index=available_years, columns=available_years)
            output_file = os.path.join(month_folder, f"{dept.replace(' ','_')}_md_{month}.csv")
            diff_df.to_csv(output_file)
            print(f"✅ Guardado: {output_file}")

    print("Todas las matrices mensuales por departamento generadas.")

if __name__ == "__main__":
    generar_cdc_departamento_mes()
