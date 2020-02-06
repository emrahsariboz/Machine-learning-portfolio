# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:04:39 2020

@author: EmrahSariboz
"""

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



load_digits = load_digits()

X = load_digits.data
y = load_digits.target

print(X.shape)
print(y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

classifier = Perceptron(max_iter=40, eta0 = 0.1, random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
