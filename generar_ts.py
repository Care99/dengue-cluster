import pandas as pd
import os

def folder():
    # Definir rutas
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_directory, 'csv/casos.csv')  # archivo de entrada
    output_folder = os.path.join(script_directory, 'csv')   # carpeta de salida
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'dengue_ts.csv')

    # Leer CSV
    data = pd.read_csv(csv_path)

    # Filtrar solo dengue y casos totales
    data = data[(data['disease'].str.contains("DENGUE")) & (data['classification'] == "TOTAL")].copy()

    # Columna a usar: incidence
    incidence_col = 'incidence'

    # Convertir columna date a tipo datetime
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    # Extraer semana y año calendario
    data['week'] = data['date'].dt.isocalendar().week
    data['year'] = data['date'].dt.year

    # Definir semana de inicio de año pandémico
    pandemic_start_week = 39

    # Asignar año pandémico:
    # Semana >= 39 → mismo año
    # Semana < 39 → año anterior
    data['Pandemic_year'] = data['year'].where(data['week'] >= pandemic_start_week, data['year'] - 1)
    
    # Ignorar datos del año 2023 por incompletos
    data = data[(data['Pandemic_year'] < 2023)]

    # Reordenar semanas: semana 39 → Week_1, ..., semana 38 → Week_52
    data['pandemic_week'] = (data['week'] - pandemic_start_week + 52) % 52 + 1

    # Pivot: una fila por departamento y año pandémico
    pivot = data.pivot_table(
        index=['name', 'Pandemic_year'],
        columns='pandemic_week',
        values=incidence_col,
        aggfunc='sum'
    ).reset_index()

    # Extraer datos de ts
    week_cols = pivot.columns[2:]

    # Convertir a numeric (NaNs stay NaN)
    pivot[week_cols] = pivot[week_cols].apply(pd.to_numeric, errors='coerce')

    # Interpolatar
    pivot[week_cols] = pivot[week_cols].interpolate(axis=1, method='linear', limit_direction='both')

    # Renombrar columna año
    pivot['Pandemic_year'] = pivot['Pandemic_year'].apply(lambda y: f"{y}-{y+1}")

    #Renombrar columna TS
    new_columns = ['Department', 'Pandemic_year']
    for w in pivot.columns[2:]:
        if w <= (52 - pandemic_start_week + 1):  # 1→14 → semanas 39→52
            new_columns.append(f'week_{w + pandemic_start_week - 1}')
        else:  # semanas 1→38
            new_columns.append(f'week_{w - (52 - pandemic_start_week + 1)}')

    pivot.columns = new_columns
    pivot.to_csv(output_file, index=False)
    print(f"CSV guardado en la carpeta 'csv': {output_file}")



def generar_ts_calendario():
    import pandas as pd
    import os

    csv_path = 'csv/casos.csv'
    output_file = 'csv/dengue_ts_historico.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Leer CSV
    data = pd.read_csv(csv_path)

    # Filtrar solo dengue y casos totales
    data = data[(data['disease'].str.contains("DENGUE")) & (data['classification'] == "TOTAL")].copy()

    # Convertir a datetime
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    # Extraer semana y año calendario
    data['week'] = data['date'].dt.isocalendar().week
    data['year'] = data['date'].dt.year

    # Solo años desde 2018 hasta 2023
    data = data[(data['year'] >= 2018) & (data['year'] <= 2023)]

    # Crear columna año_semana con semana con ceros (01, 02, …)
    data['year_week'] = data['year'].astype(str) + '_week_' + data['week'].astype(str).str.zfill(2)

    # Pivot: filas = departamento, columnas = year_week
    pivot = data.pivot_table(index='name', columns='year_week', values='incidence', aggfunc='sum').reset_index()

    # Ordenar columnas por año y número de semana
    ordered_cols = ['name'] + sorted([c for c in pivot.columns if c != 'name'],
                                     key=lambda x: (int(x.split('_')[0]), int(x.split('_week_')[1])))
    
    # Ordenar columnas por año y semana
    ordered_cols = ['name'] + sorted(
        [c for c in pivot.columns if c != 'name'],
        key=lambda x: (int(x.split('_')[0]), int(x.split('_week_')[1]))
    )
    pivot = pivot[ordered_cols]

    # Forzar conversión numérica en todas las columnas menos 'name'
    pivot.iloc[:, 1:] = pivot.iloc[:, 1:].apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Ahora hacer la interpolación
    pivot.iloc[:, 1:] = pivot.iloc[:, 1:].interpolate(axis=1, method='linear', limit_direction='both')

    # Guardar CSV
    pivot.rename(columns={'name': 'Department'}, inplace=True)
    pivot.to_csv(output_file, index=False)
    print(f"CSV histórico guardado: {output_file}")

# Ejecutar
generar_ts_calendario()
