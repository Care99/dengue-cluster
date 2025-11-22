# 1. Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Optional: TAN classifier from pyAgrum
import pyagrum.skbn as skbn

#data.csv path
csv_path = "csv"
data_csv = os.path.join(csv_path, "casos.csv")
def load_df(year,month,department):
    # 2. Load your weekly incidence dataset
    # Example structure: columns = ['city', 'week', 'cases']
    data = pd.read_csv(data_csv)
    filtered_data = data[
        (data['disease'] == "DENGUE") 
        & (data['classification'] == "TOTAL")
        & (data['name'] == department)
        ]
    last_day=[30,28,31,30,31,30,31,31,30,31,30,31]
    months = ['ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']
    start_date = "2019-10-01"
    end_date = f"{year}-{month}-{last_day[months.index(month)]}"
    filtered_data = filtered_data[filtered_data['date'].between(start_date, end_date, inclusive='both')]
    grouped = filtered_data.groupby(["name"])

    # 3. Feature engineering
    # Create lag features (cases from previous weeks)
    for lag in [1, 2, 3]:
        filtered_data[f'lag_{lag}'] = grouped['incidence'].shift(lag)

    # Rolling mean (trend indicator)
    filtered_data['rolling_mean_3'] = grouped['incidence'].transform(lambda x: x.rolling(3).mean())

    # Seasonal indicator (week of year)
    filtered_data['week_of_year'] = pd.to_datetime(df['week']).dt.isocalendar().week

    # Drop rows with NaN (due to lagging)
    #filtered_data = filtered_data.dropna()

    # 4. Define target variable
    # Example: classify incidence as "High" vs "Low"
    threshold = filtered_data['incidence'].median()
    filtered_data['target'] = (filtered_data['incidence'] > threshold).astype(int)

    X = filtered_data.drop(columns=['incidence','target','week','city'])
    y = filtered_data['target']

    # 5. Train-test split
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. Models
# CART
def CART(X_train, X_test, y_train, y_test):
    cart = DecisionTreeClassifier(max_depth=5, random_state=42)
    cart.fit(X_train, y_train)
    return cart.predict(X_test)

# Random Forest
def RANDOM_FOREST(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)

# KNN
def KNN(X_train, X_test, y_train, y_test,knn_neighbors):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    knn.fit(X_train_scaled, y_train)
    knn.predict(X_test_scaled)

# TAN (Tree Augmented Naive Bayes)
def TAN(X_train, X_test, y_train, y_test):
    tan = skbn.BNClassifier(learningMethod='TAN')
    tan.fit(X_train, y_train)
    return tan.predict(X_test)

# 7. Evaluation
def evaluate_models(year,month,department,k,n):
    X_train, X_test, y_train, y_test = load_df(year,month,department)
    cart=CART(X_train, X_test, y_train, y_test)
    rf=RANDOM_FOREST(X_train, X_test, y_train, y_test)
    knn=KNN(X_train, X_test, y_train, y_test,k*n)
    tan=TAN(X_train, X_test, y_train, y_test)
    return cart,rf,knn,tan