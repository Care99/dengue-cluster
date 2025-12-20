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
def cart_model()->SKLearnModel:
    return SKLearnModel(model=DecisionTreeRegressor(max_depth=5, random_state=42),lags=12)
def rf_model():
    return SKLearnModel(model=RandomForestRegressor(n_estimators=100, random_state=42),lags=12)