import os
from neighbors import project_time_series
from neighbors import load_time_series
from graph_errors import generate_y_x_graph
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
def load_time_series(path):
    return np.array(
        pd.read_csv(
            filepath_or_buffer=path, 
            header=None, 
            index_col=None, 
            skiprows=1
        ).to_numpy
    )
""""
Generates a set of projected time series based on the following variables:
- K: Number of nearest neighbors
- N: Number of time series retrieved for each neighbor
Returns: A set of time series for every region available in a given year
"""
def calculate_projections():
    script_directory = os.getcwd()
    processed_data_path = os.path.join(script_directory,'processed_data')
    resultado_funciones_path = os.path.join(processed_data_path,'resultado_funciones')
    departments = ['ALTO_PARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'CENTRO_EST','CENTRO_NORTE','CENTRO_SUR','CHACO','CORDILLERA',
              'METROPOLITANO','PARAGUARI','PARAGUAY','PTE_HAYES','SAN_PEDRO',
    csv_path = os.path.join(script_directory,'csv')
    resultado_funciones_path = os.path.join(csv_path,'resultado_funciones')
    departments = ['ALTO PARARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'Centro est','Centro norte','Centro sur','Chaco','CORDILLERA',
              'Metropolitano','PARAGUARI','Paraguay','PTE HAYES','SAN PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO_PARAGUAY']
    months = [9,10,11,12,1,2,3,4,5,6,7,8]
    name_of_month = [
        'OCTUBRE',
        'NOVIEMBRE',
        'DICIEMBRE',
        'ENERO',
        'FEBRERO',
        'MARZO',
        'ABRIL',
        'MAYO',
        'JUNIO',
        'JULIO',
        'AGOSTO',
        'SEPTIEMBRE'
    ]
    forecasted_values = [4,8,12,16]
    real_time_series = [
        load_time_series(
            csv_path,
            'time_series_{year}.csv',
            department
        ) 
        for department in departments
    ]
    best_rmse = 0
    best_nearest_neighbors = 0
    best_year_in_neighbor = 0
    for nearest_neighbors in range(1,25):
        for years_in_neighbor in range(1,4):
            forecast_one_month = []
            forecast_two_months = []
            forecast_three_months = []
            forecast_four_months = []
            rmse_average = 0
            for monthIndex in range(0,12,1):
                forecast_one_month.extend(
                    project_time_series(
                        nearest_neighbors,
                        years_in_neighbor,
                        months[monthIndex],
                        forecasted_values
                    )
                )
            for monthIndex in range(0,12,2):
                forecast_two_months.extend(
                    project_time_series(
                        nearest_neighbors,
                        years_in_neighbor,
                        months[monthIndex],
                        forecasted_values
                    )
                )
            for monthIndex in range(0,12,3):
                forecast_three_months.extend(
                    project_time_series(
                        nearest_neighbors,
                        years_in_neighbor,
                        months[monthIndex],
                        forecasted_values
                    )
                )
            for monthIndex in range(0,12,4):
                forecast_four_months.extend(
                    project_time_series(
                        nearest_neighbors,
                        years_in_neighbor,
                        months[monthIndex],
                        forecasted_values
                    )
                )
            error_one_month = root_mean_squared_error(forecast_one_month,real_time_series)
            error_two_months = root_mean_squared_error(forecast_two_months,real_time_series)
            error_three_months = root_mean_squared_error(forecast_three_months,real_time_series)
            error_four_months = root_mean_squared_error(forecast_four_months,real_time_series)
            generate_y_x_graph(forecast_one_month,real_time_series,nearest_neighbors,years_in_neighbor)
            generate_y_x_graph(forecast_two_months,real_time_series,nearest_neighbors,years_in_neighbor)
            generate_y_x_graph(forecast_three_months,real_time_series,nearest_neighbors,years_in_neighbor)
            generate_y_x_graph(forecast_four_months,real_time_series,nearest_neighbors,years_in_neighbor)
