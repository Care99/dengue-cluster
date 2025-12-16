from datetime import datetime, timedelta
from src.utils.constants import start_date_index, csv_path
import os
import pandas as pd
excel_name = 'casos.csv'
excel_file = os.path.join(csv_path,excel_name)
unformatted_data = pd.read_csv(excel_file)
data = unformatted_data.copy()
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
def get_time_series_from_range(start_date:datetime,end_date:datetime,department:str)->list:
    ts_dict = {}
    filtered_data = data[
        (data['disease'] == "DENGUE") 
        & (data['classification'] == "TOTAL") 
        & (data['name'] == department)]
    filtered_data = filtered_data.copy()
    filtered_data['date'] = pd.to_datetime(filtered_data['date'], format='%Y-%m-%d')
    range_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='both')]
    ts_dict[department] = range_data.reset_index(drop=True)
    return ts_dict[department]['incidence'].values.flatten().tolist()
def get_ts(year:str, week:str, department:str)->list:
    start_date = datetime.strptime(start_date_index[int(year.split('-')[0])-2019],'%Y-%m-%d') + timedelta(weeks=int(week))
    end_date = start_date + timedelta(weeks=11)
    return get_time_series_from_range(start_date,end_date,department)

def get_historical_data(department:str)->list:
    start_date = datetime.strptime("2019-10-05",'%Y-%m-%d')
    end_date = datetime.strptime("2022-09-24",'%Y-%m-%d')
    return get_time_series_from_range(start_date,end_date,department)

def get_2022_2023_data(department:str)->list:
    start_date = datetime.strptime("2022-10-01",'%Y-%m-%d')
    end_date = datetime.strptime("2023-09-30",'%Y-%m-%d')
    return get_time_series_from_range(start_date,end_date,department)