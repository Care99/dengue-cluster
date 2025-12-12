import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplt
mplt.use('SVG',force=True)
from matplotlib import rcParams
import pandas as pd
import numpy as np
from neighbors import get_2022_2023_data
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

models = [
  'naive_drift',
  'auto_arima',
  'linear_regression',
  'lstm_forecast',
  ]

classifications = [
    'CART',
    'get_cluster',
    'get_cluster_de_clusters',
    'get_cluster_jerarquico',
    'get_historical_data',
    'RANDOM_FOREST'
]
base_folder = os.getcwd()
forecast_folder = os.path.join(base_folder, 'csv', 'forecast')
def plot_dengue_forecasts(classification:str,model:str,input_department:str,month_index:str):
    """
    Plots four dengue forecast time series from different clustering methods.
    """
    original_ts = get_2022_2023_data(input_department)
    filename = f'{input_department}.csv'
    if(classification=='CART'or classification=='RANDOM_FOREST'):
        model = 'forecast_using_regression_models'
    path = os.path.join(forecast_folder,classification,model,f'{month_index}_months',filename)
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
    plt.title(f'Predictions for {input_department} using {model}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.ylim(y_min, y_max)
    # Add legend
    plt.legend(loc='best', fontsize=10, framealpha=0.9, shadow=True)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    svg_folder = os.path.join(base_folder, 'svg', 'forecast', classification, model, f'{month_index}_months')
    os.makedirs(svg_folder, exist_ok=True)
    filename = f'{input_department}_graphs.svg'
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
        for forecast in [1,2,3,4]:
            for model in models:        
                classification_values = []
                for classification in classifications:
                    department_values = []
                    for department in departments:
                        filename = f'{department}_{error_type}.txt'
                        model_name=model
                        if(classification=='CART'or classification=='RANDOM_FOREST'):
                            model_name='forecast_using_regression_models'                            
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
                plt.title(f'Error Analysis for {model} - {error_type} - {forecast} months', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Departments', fontsize=12)
                plt.ylabel('Error Value', fontsize=12)
                plt.xticks(rotation=90)
                plt.legend(loc='best', fontsize=10, framealpha=0.9, shadow=True)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                svg_folder = os.path.join(base_folder, 'svg', 'error_analysis')
                os.makedirs(svg_folder, exist_ok=True)
                filename = f'{model}_{error_type}_{forecast}_months_graphs.svg'
                svg_path = os.path.join(svg_folder, filename)
                plt.savefig(svg_path, format='svg')
                print('Saved plot to: ',svg_path)
                plt.close()
                
for classification in classifications:
    for model in models:
        for department in departments:
            for month_index in ['1','2','3','4']:
                plot_dengue_forecasts(classification,model,department,month_index)
plot_error()