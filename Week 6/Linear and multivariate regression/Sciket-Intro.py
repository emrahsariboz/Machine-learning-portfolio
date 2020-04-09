# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:32:43 2020

@author: EmrahSariboz
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


#Split X and y to the train and testing dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclasfficed samples : %d' %(y_test != y_pred).sum())
print('MSE %.2f' % mean_squared_error(y_test, y_pred))


