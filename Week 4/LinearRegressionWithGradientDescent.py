# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:46:18 2020

@author: EmrahSariboz
"""

import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sympy as sym




# =============================================================================
# X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target)
# 
# 
# clf = LinearRegression()
# clf.fit(X_train, y_train)
# 
# 
# 
# predicted = clf.predict(X_test)
# expected = y_test
# plt.figure(figsize=(4, 3))
# plt.scatter(expected, predicted)
# plt.axis('tight')
# plt.xlabel('True price ($1000s)')
# plt.ylabel('Predicted price ($1000s)')
# plt.tight_layout()
# 
# MSE =(1 / (2 * len(dataset.target)))* np.sum((predicted-expected)** 2)
# print('MSE : ', MSE)
# 
# =============================================================================

X= [2,4,5]
y = [1.2, 2.8, 5.3]

X = np.array(X).reshape(-1,1)
y = np.array(y).reshape(-1,1)
lm = LinearRegression()
lm.fit(X, y)



cost = b0 = 0
b1 = 1
error = []
lr = 0.001

for i in range(10000):
    y_pred = (b0 + b1*X)
    cost = (y - y_pred)**2
    partial_b0 = -2 * (y - (b0 + b1 * X))
    partial_b1 = (-2 * X) * (y - (b0 + b1 * X))
    b0 = b0 - lr * partial_b0.sum()
    b1 = b1 - lr * partial_b1.sum()
    error.append(cost.sum())
    

y_pred = lm.predict(X)
plt.scatter(X, y, color = 'r')
plt.plot(X, y_pred)

# =============================================================================
# plt.figure(figsize=(12, 7))
# plt.title('Gradient Descent - error')
# plt.plot(np.arange(1, len(error) + 1), error)
# plt.show()
# =============================================================================

print('B0 ' , b0, 'B1 ', b1)
prediction = b0 + b1 * 7.2



