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

cart_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
cart_model = SKLearnModel(model=cart_regressor,lags=12)

rf_rgressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model = SKLearnModel(model=rf_rgressor,lags=12)

tan_model = BNClassifier(learningMethod='TAN')

# 6. Models
# CART
# Classifies data by learning a series of decision rules from the features
def CART()->SKLearnModel:
    return cart_model

# Random Forest
# Ensemble method built on top of decision trees
def RANDOM_FOREST()->SKLearnModel:
    return rf_model

# TAN (Tree Augmented Naive Bayes)
# Builds a Bayesian Network from and uses it for classification tasks.
def TAN(time_series:TimeSeries,forecast_values:int)->BNClassifier:
    return tan_model

def fill_na(time_series):
    series = pd.Series(time_series)
    return series.fillna(series.mean()).tolist()