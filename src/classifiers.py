# 1. Import libraries
import os
import pandas as pd
from darts.models import RegressionModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from darts import TimeSeries
# Optional: TAN classifier from pyAgrum
from pyagrum.skbn import BNClassifier

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