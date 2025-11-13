import os
import pandas as pd
import numpy as np
import matplotlib as mplt; mplt.use('SVG',force=True)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
departments = ['ALTO PARARANA','AMAMBAY','ASUNCION','CAAGUAZU','CENTRAL',
              'Centro est','Centro norte','Centro sur','Chaco','CORDILLERA',
              'Metropolitano','PARAGUARI','Paraguay','PTE HAYES','SAN PEDRO',
              'CANINDEYU','CONCEPCION','ITAPUA','MISIONES','BOQUERON','GUAIRA',
              'CAAZAPA','NEEMBUCU','ALTO PARAGUAY']
time_series_2022 = '2022'
cluster_clusters = '_cpto'
historical = 'historical_time_series'
knn = 'historical_time_series'


def scatter_actual_vs_projected(
    actual,
    projected,
    ax=None,
    title="Actual vs Projected",
    xlabel="Actual",
    ylabel="Projected",
    point_color="C0",
    point_alpha=0.6,
    point_size=30,
    identity_color="k",
    identity_linestyle="--",
    annotate_metrics=True,
    show_legend=False,
    tolerance=None,        # e.g., 0.1 for ±10% band or absolute if use_relative=False
    tolerance_relative=True,
    figsize=(6,6),
    marker="o"
):
    """
    Plot actual vs projected scatter with a y=x line.
    actual, projected: array-like (same length)
    tolerance: if provided, draws a band around identity:
      - if tolerance_relative=True, tolerance is fractional (0.1 = 10%)
      - otherwise tolerance treated as absolute value
    Returns: matplotlib Axes
    """
    # convert
    actual = pd.Series(actual).astype(float).reset_index(drop=True)
    projected = pd.Series(projected).astype(float).reset_index(drop=True)
    if len(actual) != len(projected):
        raise ValueError("actual and projected must have the same length")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # scatter
    ax.scatter(actual, projected, c=point_color, alpha=point_alpha, s=point_size, marker=marker, label="points" if show_legend else None)

    # identity line limits
    mn = min(actual.min(), projected.min())
    mx = max(actual.max(), projected.max())
    padding = 0.02 * (mx - mn) if mx > mn else 1.0
    x_vals = np.array([mn - padding, mx + padding])
    ax.plot(x_vals, x_vals, color=identity_color, linestyle=identity_linestyle, label="y = x" if show_legend else None)

    # tolerance band
    if tolerance is not None:
        if tolerance_relative:
            lower = x_vals * (1 - tolerance)
            upper = x_vals * (1 + tolerance)
        else:
            lower = x_vals - tolerance
            upper = x_vals + tolerance
        ax.fill_between(x_vals, lower, upper, color=identity_color, alpha=0.08)

    # metrics
    if annotate_metrics:
        y_true = actual.values
        y_pred = projected.values
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = np.nan
        text = f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # labels and tidy
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(mn - padding, mx + padding)
    ax.set_ylim(mn - padding, mx + padding)
    ax.grid(True, linestyle=":", linewidth=0.5)
    if show_legend:
        ax.legend()
    return ax
    
for i in range(1,25):
    region = str(i)
    if(i<=9):
        region = '0'+str(i)
    plt.figure()
    x = list(range(1,53))
    time_series1 = pd.read_csv(f'2022_{region}.csv', header=None)
    y = time_series1.iloc[0,:].values
    plt.plot(x,y,'b',label='Datos reales')
    plt.legend()
    time_series2 = pd.read_csv(f'cluster_clusters_time_series_cpto{region}.csv', header=None)
    y = time_series2.iloc[0,:].values
    plt.plot(x,y,'g',label='CC_CPTO')
    plt.legend()
    time_series3 = pd.read_csv(f'cluster_clusters_time_series_sarima{region}.csv', header=None)
    y = time_series3.iloc[0,:].values
    plt.plot(x,y,'r',label='CC_SARIMA')
    plt.legend()
    time_series4 = pd.read_csv(f'historical_time_series_cpto{region}.csv', header=None)
    y = time_series4.iloc[0,:].values
    plt.plot(x,y,'c--',label='H_CPTO')
    plt.legend()
    time_series5 = pd.read_csv(f'historical_time_series_sarima{region}.csv', header=None)
    y = time_series5.iloc[0,:].values
    plt.plot(x,y,'m--',label='H_SARIMA')
    plt.legend()
    time_series6 = pd.read_csv(f'knn_time_series_cpto{region}.csv', header=None)
    y = time_series6.iloc[0,:].values
    plt.plot(x,y,'y--',label='KNN_CPTO')
    plt.legend()
    time_series7 = pd.read_csv(f'knn_time_series_sarima{region}.csv', header=None)
    y = time_series7.iloc[0,:].values
    plt.plot(x,y,'k--',label='KNN_SARIMA')
    plt.legend()
    plt.xlabel("Semanas")
    plt.ylabel("Incidencia")
    plt.title(f'{departments[i-1]}')
    plt.grid(True)

    # Save plot to SVG
    plt.savefig(f'{departments[i-1]}.svg', format="svg")
    plt.close()
    print(f'Plot saved as {departments[i-1]}.svg')