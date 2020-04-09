# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:13:54 2020

@author: EmrahSariboz
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

lr = LinearRegression()

lr.fit(X_train, y_train)

prediction = lr.predict(X_test)

#Visualization The Training set result

plt.scatter(X_train, y_train)
plt.plot(X_train, lr.predict(X_train), color = 'r')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
plt.title("Visualization using Training set")


plt.title("Visualization using Test set")
plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color = 'r')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()


print('MSE %.2f' % mean_squared_error(y_test, prediction))
