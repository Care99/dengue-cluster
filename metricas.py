#step 1
import pandas as pd
import os
import matplotlib as mplt; mplt.use('SVG',force=True)
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram,linkage
import numpy as np
import math
from pandas import DataFrame
script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
ts_historico_path = os.path.join(csv_path,'ts_historico')

svg_path = os.path.join(script_directory,'svg')
months = ['ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']
years = [2019,2020,2021,2022,2023]
departments = ['ALTO_PARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'CENTRO_EST','CENTRO_NORTE','CENTRO_SUR','CHACO','CORDILLERA',
              'METROPOLITANO','PARAGUARI','PARAGUAY','PTE_HAYES','SAN_PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO_PARAGUAY']
def folder(year,month,department):
  # Read the Excel file
  excel_name = 'casos.csv'
  excel_file = os.path.join(csv_path,excel_name)
  data = pd.read_csv(excel_file)

  # Apply the filter
  # Create a folder with the specified format
  os.makedirs(ts_historico_path,exist_ok=True)
  year_path = os.path.join(ts_historico_path,str(year))
  os.makedirs(year_path,exist_ok=True)
  month_path = os.path.join(year_path,month)
  os.makedirs(month_path,exist_ok=True)

  last_day=[30,28,31,30,31,30,31,31,30,31,30,31]
  # Manage the time ranges
  # Iterate through filtered data and create subfolders
  # Initialize an empty DataFrame to store incidence data for the year
  incidence_data = pd.DataFrame()
  # Filter the data
  filtered_data = data[
     (data['disease'] == "DENGUE") 
     & (data['classification'] == "TOTAL") 
     & (data['name'] == department)]

  # Convert date column to pd.date
  filtered_data = filtered_data.copy()
  filtered_data['date'] = pd.to_datetime(filtered_data['date'], format='%Y-%m-%d')

  # Define the start and end dates
  start_date = pd.to_datetime(f"{year}-{months.index(month)+1}-1", format='%Y-%m-%d')
  end_day = last_day[months.index(month)]
  if( month == 'FEBRERO' and year == 2020 ):
    end_day = 29
  end_date = pd.to_datetime(f"{year}-{months.index(month)+1}-{end_day}", format='%Y-%m-%d')

  # Filter the data for the specified period
  range_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='both')]

  # Extract the incidence columns
  incidence_columns = [col for col in range_data.columns if 'incidence' in col]
  incidence_values = range_data[incidence_columns].values.flatten()  # Get incidence values as a flat array

  # Create a row with the department name and incidence values
  row_data = incidence_values.tolist()
  incidence_data = pd.DataFrame([row_data])

  # Save the DataFrame as a CSV file
  output_file_name = f'{department}.csv'

  output_file = os.path.join(month_path, output_file_name)
  incidence_data.to_csv(output_file, header=False, index=False)
  print(f"Saved: csv/{year}/{month}/{output_file_name}")

for year in years:
   for month in months:
      for department in departments:
         folder(year,month,department)