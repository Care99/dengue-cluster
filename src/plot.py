import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplt
mplt.use('SVG',force=True)
from matplotlib import rcParams
import pandas as pd
import numpy as np
from src.utils.time_series import get_2022_2023_data
import os
from utils.constants import departments
from darts import TimeSeries
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
def plot_scatter(
    actual:TimeSeries,
    predicted:TimeSeries,
    input_department:str,
    model:str,
    classification:str,
    weeks_to_forecast:int
  )->None:
  actual = actual.values().flatten()
  predicted = predicted.values().flatten()
  plt.figure(figsize=(10, 6))
  # Calculate regression line for reference
  z = np.polyfit(actual, predicted, 1)
  p = np.poly1d(z)
  x_line = np.linspace(min(actual), max(actual), 100)
  y_line = p(x_line)
  
  # Create color gradient based on prediction error
  errors = np.abs(predicted - actual)
  normalized_errors = (errors - errors.min()) / (errors.max() - errors.min())
  plt.scatter(
    actual,
    predicted,
    c=normalized_errors,
    cmap='viridis',
    alpha=0.7,
    s=50 + 100 * normalized_errors,  # Size varies with error
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
  plt.title(
      f'Scatter Plot: {input_department}\n'
      f'Model: {model} | Classification: {classification} | Horizon: {weeks_to_forecast} months',
      fontsize=14, fontweight='bold', pad=20
  )
  plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
  plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
  r2 = np.corrcoef(actual, predicted)[0, 1]**2
  plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
  plt.legend()
  path = os.path.join(csv_path,'forecast',classification,model,f'{weeks_to_forecast}_months')
  os.makedirs(path,exist_ok=True)
  output_file_name = f'{input_department}_scatter.svg'
  output_file = os.path.join(path, output_file_name)
  plt.savefig(output_file)
  plt.close()
  print(f"Saved: {output_file}")
def plot_histogram(
    actual: TimeSeries,
    predicted: TimeSeries,
    input_department: str,
    model: str,
    classification: str,
    weeks_to_forecast: int
) -> None:
    # Extract data
    actual_vals = actual.values().flatten()
    predicted_vals = predicted.values().flatten()
    
    # Calculate IQR for actual values to determine reasonable bounds
    actual_q1 = np.percentile(actual_vals, 25)
    actual_q3 = np.percentile(actual_vals, 75)
    actual_iqr = actual_q3 - actual_q1
    
    # Define bounds based on actual values (using 1.5*IQR rule or actual min/max)
    lower_bound = max(predicted_vals.min(), actual_vals.min())
    upper_bound = min(predicted_vals.max(), actual_q3 + 1.5 * actual_iqr)
    
    # Filter predicted values for visualization (but keep all for stats)
    predicted_filtered = predicted_vals[(predicted_vals >= lower_bound) & 
                                       (predicted_vals <= upper_bound)]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    # Use bins based on actual values range
    bins = np.histogram_bin_edges(actual_vals, bins='auto')
    
    # --- Plot 1: Zoomed-in view (using actual value range) ---
    ax1.hist(predicted_vals, bins=bins, alpha=0.6, label='Predicted',
             color='#FF6B6B', edgecolor='black', linewidth=0.5,
             range=(lower_bound, upper_bound))
    ax1.hist(actual_vals, bins=bins, alpha=0.6, label='Actual',
             color='#4ECDC4', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Zoomed View (Actual Value Range)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add outlier info
    outlier_count = len(predicted_vals) - len(predicted_filtered)
    outlier_pct = (outlier_count / len(predicted_vals)) * 100
    info_text = f'Focus Range: [{lower_bound:.1f}, {upper_bound:.1f}]\n' \
                f'Outliers excluded: {outlier_count} ({outlier_pct:.1f}%)'
    ax1.text(0.08, 0.98, info_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # --- Plot 2: Box plot for outlier visualization ---
    data_to_plot = [actual_vals, predicted_vals]
    box_colors = ['#4ECDC4', '#FF6B6B']
    
    bp = ax2.boxplot(data_to_plot, patch_artist=True, labels=['Actual', 'Predicted'],
                     showfliers=True, whis=1.5)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Color the medians
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    
    # Add individual points for actual values (in background)
    for i, (data, color) in enumerate(zip(data_to_plot, box_colors), 1):
        # Add jitter to x-coordinate
        x = np.random.normal(i, 0.04, size=len(data))
        ax2.scatter(x, data, alpha=0.3, color=color, s=20)
    
    ax2.set_ylabel('Box Plot with Outliers', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add statistics as text
    actual_stats = f'Actual: μ={np.mean(actual_vals):.1f}, σ={np.std(actual_vals):.1f}'
    pred_stats = f'Predicted: μ={np.mean(predicted_vals):.1f}, σ={np.std(predicted_vals):.1f}'
    ax2.text(0.98, 0.98, f'{actual_stats}\n{pred_stats}', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Main title
    fig.suptitle(
        f'Distribution Analysis with Outlier Handling: {input_department}\n'
        f'Model: {model} | Classification: {classification} | Horizon: {weeks_to_forecast} months',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    # Save
    path = os.path.join(csv_path, 'forecast', classification, model, f'{weeks_to_forecast}_months')
    os.makedirs(path, exist_ok=True)
    output_file_name = f'{input_department}_hist.svg'
    output_file = os.path.join(path, output_file_name)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")
    print(f"Outlier info: {outlier_count} predicted values ({outlier_pct:.1f}%) excluded from histogram view")
for classification in classifications:
    for model in models:
        for department in departments:
            for month_index in ['1','2','3','4']:
                plot_dengue_forecasts(classification,model,department,month_index)
plot_error()