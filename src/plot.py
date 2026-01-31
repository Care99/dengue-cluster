import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplt
mplt.use('SVG',force=True)
from matplotlib import rcParams
#mplt.rcParams['font.family'] = 'serif'
#mplt.rcParams['font.serif'] = 'Times New Roman'
mplt.rcParams['font.weight'] = 'normal'  # Correct spelling
mplt.rcParams['axes.labelweight'] = 'normal'
mplt.rcParams['axes.titleweight'] = 'normal'
mplt.rcParams['font.size'] = 8
import pandas as pd
import numpy as np
from src.utils.time_series import get_2022_2023_data
import os
from src.utils.constants import departments, csv_path
from darts import TimeSeries
models = [
  'NAIVE_MODEL',
  'AUTO_ARIMA',
  'LINEAR_REGRESSION',
  'LSTM',
  ]

classifications = [
    'CART',
    'get_cluster',
    'get_cluster_de_clusters',
    'get_cluster_jerarquico',
    'HISTORICAL_DATA',
    'RANDOM_FOREST'
]
base_folder = os.getcwd()
forecast_folder = os.path.join(base_folder, 'csv', 'forecast')
def plot_dengue_forecasts():
    for forecast in [1,2,3,4,53]:
            for model in models:        
                classification_values = []
                for classification in classifications:
                    department_values = []
                    for department in departments:
                        """
                        Plots four dengue forecast time series from different clustering methods.
                        """
                        original_ts = get_2022_2023_data(department)
                        filename = f'{department}.csv'
                        model_name = model
                        if(classification=='CART'or classification=='RANDOM_FOREST'):
                            model_name = 'state_of_art'
                        path = os.path.join(forecast_folder,classification,model_name,f'{forecast}_months',filename)
                        predicted_ts = pd.read_csv(path,header=None)[0].tolist()

                        original_min = np.min(original_ts)
                        original_max = np.max(original_ts)
                        original_range = original_max - original_min
                        
                        # Add 20% margin to the range
                        margin = original_range * 0.2
                        y_min = max(0, original_min - margin)  # Don't go below 0 for dengue cases
                        y_max = original_max + margin
                        
                        # Time series arrays
                        time_series = [original_ts, predicted_ts]
                        labels = ['Original TS', 'Predicted TS']
                        # Colors for each series
                        colors = ['#FF0000','#0000FF']
                        
                        # Line styles for better differentiation
                        line_styles = ['--','-']
                        
                        x_values = range(1,54) # Assuming all series have the same length for x-axis
                        
                        # Plot each time series
                        for i, (ts, label, color, line_style) in enumerate(zip(time_series, labels, colors, line_styles)):
                            # Ensure ts is a numpy array for consistency
                            ts_array = np.array(ts)
                            
                            # Create x-axis values for this specific series
                            if len(x_values) != len(ts_array):
                                x_for_ts = range(len(ts_array))
                            else:
                                x_for_ts = x_values[:len(ts_array)]
                            
                            # Plot with styling
                            plt.figure(figsize=(3.15, 3.15))
                            plt.plot(x_for_ts, ts_array, 
                                    color=color, 
                                    linestyle=line_style,
                                    linewidth=2,
                                    markerfacecolor=color,
                                    markeredgecolor='white',
                                    markeredgewidth=1,
                                    label=label,
                                    alpha=0.8)
                        
                        # Customize plot
                        plt.xlabel('Time Period', fontsize=8)
                        plt.ylabel('Value', fontsize=8)
                        plt.ylim(y_min, y_max)
                        # Add legend
                        plt.legend(loc='best', fontsize=8, framealpha=0.9, shadow=True)
                        
                        # Add grid
                        plt.grid(True, alpha=0.3, linestyle='--')
                        
                        # Tight layout
                        plt.tight_layout()
                        
                        svg_folder = os.path.join(base_folder, 'svg', 'forecast', classification, model_name, f'{forecast}_months')
                        os.makedirs(svg_folder, exist_ok=True)
                        filename = f'{department}_graphs.svg'
                        svg_path = os.path.join(svg_folder, filename)
                        plt.savefig(svg_path, format='svg')
                        print('Saved plot to: ',svg_path)
                        plt.close()

def plot_error():
    base_folder = os.getcwd()
    error_folder = os.path.join(base_folder, 'csv', 'error_analysis')
    """
    Plots four dengue forecast time series from different clustering methods.
    """
    for error_type in ['mae','rmse','smape']:
        for forecast in [1,2,3,4,53]:
            for model in models:        
                classification_values = []
                for classification in classifications:
                    department_values = []
                    for department in departments:
                        filename = f'{department}_{error_type}.txt'
                        model_name=model
                        if(classification=='CART'or classification=='RANDOM_FOREST'):
                            model_name='state_of_art'                            
                        path = os.path.join(forecast_folder,classification,model_name,f'{forecast}_months',filename)
                        #Read txt file
                        with open(path, 'r') as file:
                            error_value = float(file.read().strip())
                        department_values.append(error_value)
                    classification_values.append(department_values)
                argsorted = np.argsort(classification_values[2])
                #Reverse argsorted
                argsorted = argsorted[::-1]
                max_y=0
                for i in range(len(classification_values)):
                    classification_values[i] = np.array(classification_values[i])
                    #apply argsorted
                    classification_values[i] = classification_values[i][argsorted]
                    max_y = max(max_y, np.max(classification_values[i]))
                
                if(error_type=='smape'):
                    max_y = min(max_y, 200)
                else:
                    max_y = min(max_y, 100)
                plt.figure(figsize=(12, 6))
                #Choose hex color for six classifications
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                #plt.bar for each array in classification_values
                for i,(label,color) in enumerate(zip(classifications,colors)):
                    x=np.array(departments)[argsorted]
                    y=classification_values[i]
                    #Use plt.plot with argsorted
                    plt.figure(figsize=(3.15, 3.15))
                    plt.plot(
                        x, 
                        y,
                        label=label, 
                        color=color,
                        linestyle='-', 
                        linewidth=2, 
                        markersize=6, 
                        alpha=0.8
                    )
                plt.ylim(0, max_y*1.1)
                #Customize plot
                plt.xlabel('Departments', fontsize=8)
                plt.ylabel('Error Value', fontsize=8)
                plt.xticks(rotation=90)
                plt.legend(loc='best', fontsize=8, framealpha=0.9, shadow=True)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                svg_folder = os.path.join(base_folder, 'svg', 'error_analysis')
                os.makedirs(svg_folder, exist_ok=True)
                filename = f'{model}_{error_type}_{forecast}_months_graphs.svg'
                svg_path = os.path.join(svg_folder, filename)
                plt.savefig(svg_path, format='svg')
                print('Saved plot to: ',svg_path)
                dataframe=pd.DataFrame(data=classification_values,columns=np.array(departments)[argsorted])
                filename = f'{model}_{error_type}_{forecast}_months_graphs.csv'
                svg_path = os.path.join(svg_folder, filename)
                dataframe.to_csv(svg_path)
                print('Saved csv to: ',svg_path)
                plt.close()
def plot_paraguay_error():
    base_folder = os.getcwd()
    classifications_values = []
    classification_values = []
    error_types = []
    error_type='mae'
    for classification in classifications:
        classification_values = []
        for department in departments:
            forecast=1
            filename = f'{department}_{error_type}.txt'
            if(classification=='CART'or classification=='RANDOM_FOREST'):
                model_name='state_of_art'                            
            else:
                model_name='AUTO_ARIMA'
            path = os.path.join(forecast_folder,classification,model_name,f'{forecast}_months',filename)
            #Read txt file
            with open(path, 'r') as file:
                error_value = float(file.read().strip())
            classification_values.append(error_value)
        classifications_values.append(classification_values)
    print(classifications_values)
    dataframe=pd.DataFrame(data=classifications_values,columns=departments)
    filename = f'mae_benchmark.csv'
    svg_folder = os.path.join(base_folder, 'svg', 'error_analysis')
    svg_path = os.path.join(svg_folder, filename)
    dataframe.to_csv(svg_path)
    print('Saved csv to: ',svg_path)
def plot_scatter(
    actual_time_series:TimeSeries,
    predicted_time_series:TimeSeries,
    input_department:str,
    model:str,
    classification:str,
    weeks_to_forecast:int
  )->None:
  actual = actual_time_series.values().flatten()
  predicted = predicted_time_series.values().flatten()
  #plt.figure(figsize=(10, 6))
  # Calculate regression line for reference
  z = np.polyfit(actual, predicted, 1)
  p = np.poly1d(z)
  x_line = np.linspace(min(actual), max(actual), 100)
  y_line = p(x_line)
  
  # Create color gradient based on prediction error
  errors = np.abs(predicted - actual)
  normalized_errors = (errors - errors.min()) / (errors.max() - errors.min())
  plt.figure(figsize=(3.15, 3.15))
  plt.scatter(
    actual,
    predicted,
    c=normalized_errors,
    cmap='viridis',
    alpha=0.7,
    s=100 * normalized_errors,  # Size varies with error
  )
  # Perfect prediction line (y = x)
  min_val = actual.min()
  max_val = actual.max()
  plt.plot(
    [min_val, max_val], 
    [min_val, max_val], 
    'r--', 
    alpha=0.8, 
    linewidth=2
  )
  # Regression line
  plt.plot(
    x_line, 
    y_line, 
    'orange', 
    alpha=0.8, 
    linewidth=2, 
    label=f'Regression (slope={z[0]:.3f})'
  )    
  # Title and labels with better formatting
  plt.xlabel('Expected Values', fontsize=8)
  plt.ylabel('Observed Values', fontsize=8)
  r2 = np.corrcoef(actual, predicted)[0, 1]**2
  plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
  plt.legend()
  path = os.path.join(csv_path,'forecast',classification,model,f'{weeks_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file_name = f'{input_department}_scatter.svg'
  output_file = os.path.join(path, output_file_name)
  plt.savefig(output_file)
  plt.close()
  print(f"Saved: {output_file}")
def plot_errorbar(
    actual: TimeSeries,
    predicted: TimeSeries,
    input_department: str,
    model: str,
    classification: str,
    weeks_to_forecast: int
) -> None:
    actual_time_series = actual.values().flatten()
    predicted_time_series = predicted.values().flatten()
    # Matplotlib function that graphs an error bar for two numpy arrays: actual_time_series and predicted_time_series
    errors = np.abs(predicted_time_series - actual_time_series)
    plt.figure(figsize=(3.15, 3.15))
    plt.errorbar(
        range(len(actual_time_series)),
        predicted_time_series,
        yerr=errors,
        fmt='o',
        ecolor='lightgray',
        elinewidth=3,
        capsize=0,
        alpha=0.7
    )
    plt.plot(actual_time_series, 'r-', label='Actual Values', alpha=0.8)
    plt.xlabel('Time Period', fontsize=8)
    plt.ylabel('Values', fontsize=8)
    plt.legend()
    # Save
    path = os.path.join(csv_path, 'forecast', classification, model, f'{weeks_to_forecast}_months')
    os.makedirs(path, exist_ok=True)
    output_file_name = f'{input_department}_errorbar.svg'
    output_file = os.path.join(path, output_file_name)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
actual_time_series = TimeSeries.from_values(np.array(get_2022_2023_data('PARAGUAY')))
predicted_time_series = TimeSeries.from_values(np.array(pd.read_csv(os.path.join(forecast_folder,'get_cluster_de_clusters','AUTO_ARIMA','1_months','PARAGUAY.csv'),header=None)[0].tolist()))
print(actual_time_series.values().flatten())
print(len(actual_time_series))
print(predicted_time_series.values().flatten())
print(len(predicted_time_series))
input_department='PARAGUAY'
model='AUTO_ARIMA'
classification='get_cluster_de_clusters'
weeks_to_forecast=1
plot_scatter(
    actual_time_series,
    predicted_time_series,
    input_department,
    model,
    classification,
    weeks_to_forecast
  )
plot_errorbar(
    actual=actual_time_series,
    predicted=predicted_time_series,
    input_department=input_department,
    model=model,
    classification=classification,
    weeks_to_forecast=1
)
#plot_paraguay_error()
#plot_dengue_forecasts()
#plot_error()