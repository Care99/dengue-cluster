import os
departments = [
  'ALTO_PARANA',
  'AMAMBAY',
  'ASUNCION',
  'CAAGUAZU',
  'CENTRAL',
  'CENTRO_EST',
  'CENTRO_NORTE',
  'CENTRO_SUR',
  'CHACO',
  'CORDILLERA',
  'METROPOLITANO',
  'PARAGUARI',
  'PARAGUAY',
  'PTE_HAYES',
  'SAN_PEDRO',
  'CANINDEYU',
  'CONCEPCION',
  'ITAPUA',
  'MISIONES',
  'BOQUERON',
  'GUAIRA',
  'CAAZAPA',
  'NEEMBUCU',
  'ALTO_PARAGUAY'
  ]
start_date_index =["07/13/2019","07/04/2020","07/10/2021","07/09/2022"]
script_directory = os.getcwd()
csv_path = os.path.join(script_directory,'csv')
time_series_2022_2023_length=53
time_series_window=12