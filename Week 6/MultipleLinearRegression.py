# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:45:46 2020

@author: EmrahSariboz
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

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

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_prediction = regressor.predict(X_test)

print("Before backward elimination" , regressor.score(X_test,y_test))




#Building the optimal model using Backward Elimination
#Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X , axis = 1)
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_optimal).fit()




def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y,x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary() 
    return x
    

SL = 0.05
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
X_modeled = backwardElimination(X_optimal, SL)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_prediction = regressor.predict(X_test)

print("After backward elimination" , regressor.score(X_test,y_test))