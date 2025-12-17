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
class classifiers():
    def __init__(self):
        self.cart_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
        self.cart_model = SKLearnModel(model=self.cart_regressor,lags=12)

        self.rf_rgressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model = SKLearnModel(model=self.rf_rgressor,lags=12)

        self.tan_model = BNClassifier(learningMethod='TAN')

    # 6. Models
    # CART
    # Classifies data by learning a series of decision rules from the features
    def CART(self)->SKLearnModel:
        return self.cart_model

    # Random Forest
    # Ensemble method built on top of decision trees
    def RANDOM_FOREST(self)->SKLearnModel:
        return self.rf_model

    # TAN (Tree Augmented Naive Bayes)
    # Builds a Bayesian Network from and uses it for classification tasks.
    def TAN(self)->BNClassifier:
        return self.tan_model