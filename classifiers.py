# 1. Import libraries
import os
import pandas as pd
import numpy as np
from darts.models import RegressionModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from darts import TimeSeries
# Optional: TAN classifier from pyAgrum
from pyagrum.skbn import BNClassifier
#data.csv path
csv_path = "csv"
data_csv = os.path.join(csv_path, "casos.csv")
def load_df(time_series):
    data = pd.DataFrame(time_series,columns=['incidence'])
    for lag in [1, 2, 3]:
        data[f'lag_{lag}'] = data['incidence'].shift(lag)

    # Rolling mean (trend indicator)
    data['rolling_mean_3'] = data['incidence'].transform(lambda x: x.rolling(3).mean())

    # Seasonal indicator (week of year)
    #filtered_data['week_of_year'] = pd.to_datetime(df['week']).dt.isocalendar().week

    # Drop rows with NaN (due to lagging)
    #filtered_data = filtered_data.dropna()

    # 4. Define target variable
    # Example: classify incidence as "High" vs "Low"
    threshold = data['incidence'].median()
    data['target'] = (data['incidence'] > threshold).astype(int)

    X = data.drop(columns=["target"])
    y = data['target']

    # 5. Train-test split
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. Models
# CART
# Classifies data by learning a series of decision rules from the features
def CART(time_series:TimeSeries,forecast_values:int):
    cart = DecisionTreeRegressor(max_depth=5, random_state=42)
    model = RegressionModel(model=cart,lags=12)
    model.fit(time_series)
    return model.predict(forecast_values)

# Random Forest
# Ensemble method built on top of decision trees
def RANDOM_FOREST(time_series:TimeSeries,forecast_values:int):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model = RegressionModel(model=rf,lags=12)
    model.fit(time_series)
    return model.predict(forecast_values)


# TAN (Tree Augmented Naive Bayes)
# Builds a Bayesian Network from and uses it for classification tasks.
def TAN(time_series:TimeSeries,forecast_values:int):
    time_series = time_series.values().flatten()
    tan = BNClassifier(learningMethod='TAN')
    tan.fit(time_series)
    return tan.predict(forecast_values)


def fill_na(time_series):
    series = pd.Series(time_series)
    return series.fillna(series.mean()).tolist()