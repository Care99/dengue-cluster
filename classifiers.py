# 1. Import libraries
import os
import pandas as pd
import numpy as np
from darts.models import SKLearnClassifierModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Optional: TAN classifier from pyAgrum
import pyAgrum.skbn as skbn

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
def CART(X_train, X_test, y_train, y_test):
    cart = DecisionTreeClassifier(max_depth=5, random_state=42)
    cart.fit(X_train, y_train)
    return cart.predict(X_test)

# Random Forest
# Ensemble method built on top of decision trees
def RANDOM_FOREST(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


# TAN (Tree Augmented Naive Bayes)
# Builds a Bayesian Network from and uses it for classification tasks.
def TAN(X_train, X_test, y_train, y_test):
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    tan = skbn.BNClassifier(learningMethod='TAN')
    tan.fit(X_train, y_train)
    return tan.predict(X_test)


def fill_na(time_series):
    series = pd.Series(time_series)
    return series.fillna(series.mean()).tolist()

# 7. Evaluation
def evaluate_models(time_series,k,n):
    X_train, X_test, y_train, y_test = load_df(time_series)
    cart=SKLearnClassifierModel(model=CART(X_train, X_test, y_train, y_test))
    rf=SKLearnClassifierModel(model=RANDOM_FOREST(X_train, X_test, y_train, y_test))
    tan=SKLearnClassifierModel(model=TAN(X_train, X_test, y_train, y_test))
    return cart,rf,tan
