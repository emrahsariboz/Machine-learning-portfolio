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


dataset = load_boston()

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target)

clf = LinearRegression()
clf.fit(X_train, y_train)



predicted = clf.predict(X_test)
expected = y_test
plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()

MSE =(1 / (2 * len(dataset.target)))* np.sum((predicted-expected)** 2)
print('MSE : ', MSE)



X = dataset.data
y = dataset.target

def gradient_descent(X, y, eta = 0.0001, n_iter = 1000):
    '''
    Gradient descent implementation
    eta: Learning Rate
    n_iter: Number of iteration
    '''
    b0 = 0
    b1 = 0
    costList = []
    for i in range(n_iter):
        y_predict = b0 + X*b1
        y_predict = y_predict[:,0]
        b0 += -eta * (1/len(y)) * np.sum(y- y_predict)**2
        b1 += -eta * (1/len(y)) * (X * np.sum(y - y_predict)**2)
        costList.append((1 / (len(y))) * np.sum((y-y_predict)))
    return costList

gradient_descent(X, y)