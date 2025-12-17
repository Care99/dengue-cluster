# 1. Import libraries
from src.forecast import safe_log,safe_exp

import os
import pandas as pd
from darts.models import SKLearnModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from darts import TimeSeries
# Optional: TAN classifier from pyAgrum
from pyagrum.skbn import BNClassifier

# 6. Models
# CART
# Classifies data by learning a series of decision rules from the features
def CART(time_series:TimeSeries,forecast_values:int)->list[float]:
    scaled_time_series = safe_log(time_series)
    data = scaled_time_series
    cart = DecisionTreeRegressor(max_depth=5, random_state=42)
    model = SKLearnModel(model=cart,lags=12)
    model.fit(data)
    generated_scaled_time_series = model.predict(forecast_values)
    generated_time_series = safe_exp(generated_scaled_time_series)
    return generated_time_series.values().flatten().tolist()

# Random Forest
# Ensemble method built on top of decision trees
def RANDOM_FOREST(time_series:TimeSeries,forecast_values:int)->list[float]:
    scaled_time_series = safe_log(time_series)
    data = scaled_time_series
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model = SKLearnModel(model=rf,lags=12)
    model.fit(data)
    generated_scaled_time_series = model.predict(forecast_values)
    generated_time_series = safe_exp(generated_scaled_time_series)
    return generated_time_series.values().flatten().tolist()


# TAN (Tree Augmented Naive Bayes)
# Builds a Bayesian Network from and uses it for classification tasks.
def TAN(time_series:TimeSeries,forecast_values:int)->list[float]:
    scaled_time_series = safe_log(time_series)
    data = scaled_time_series
    tan = BNClassifier(learningMethod='TAN')
    tan.fit(data)
    generated_scaled_time_series = tan.predict(forecast_values)
    generated_time_series = safe_exp(generated_scaled_time_series)
    return generated_time_series.values().flatten().tolist()


def fill_na(time_series):
    series = pd.Series(time_series)
    return series.fillna(series.mean()).tolist()