# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:45:46 2020

@author: EmrahSariboz
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, [-1]].values


#One-hot encoding


labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap

X = X[:, 1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

