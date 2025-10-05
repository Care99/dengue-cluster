import pandas as pd
import numpy as np
import os
from glob import glob

base_folder = 'csv/matrix_diff'           # Base folder with month subfolders
output_folder = 'csv/cdc_matrix_diff'     # Output folder for combined monthly matrices
os.makedirs(output_folder, exist_ok=True)

# List month folders
month_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

for month in month_folders:
    month_path = os.path.join(base_folder, month)
    
    # Get all department CSVs in the month folder
    dept_files = glob(os.path.join(month_path, '*.csv'))
    
    df_list = []
    departments = []
    
    # Read all department CSVs into a dict
    for dept_file in dept_files:
        dept_name = os.path.basename(dept_file).split('_md_')[0]
        departments.append(dept_name)
        df = pd.read_csv(dept_file, index_col=0, dtype=str)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        df_list.append(df)
    
    n_depts = len(departments)
    combined_matrix = np.zeros((n_depts, n_depts))
    
    # Compute Bhattacharyya-like distances between departments
    for i in range(n_depts):
        for j in range(n_depts):
            if i == j:
                combined_matrix[i, j] = 0
            else:
                df_i = df_list[i]
                df_j = df_list[j]
                # Find overlapping columns (years)
                common_years = df_i.columns.intersection(df_j.columns)
                values = []
                for y in common_years:
                    try:
                        val = float(df_i.iloc[0][y]) + float(df_j.iloc[0][y])
                        values.append(val)
                    except KeyError:
                        continue
                combined_matrix[i, j] = np.mean(values) if values else 0
    
    # Save combined matrix for the month
    combined_df = pd.DataFrame(combined_matrix, index=departments, columns=departments)
    out_file = os.path.join(output_folder, f"cdc_{month}.csv")
    combined_df.to_csv(out_file)
    print(f"Saved cluster matrix for month {month}: {out_file}")
