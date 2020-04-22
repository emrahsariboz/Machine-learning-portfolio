# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:46:25 2020

@author: EmrahSariboz
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

class my_lasso_model:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def fit(self, X_train, y_train, alpha):
        print('Degree is ', alpha)
        self.lasso = Lasso(alpha=alpha)
        self.lasso.fit(X_train, y_train)
    
    def predict(self, X_train, y_train):
        return self.lasso.predict(X_train)
        
    def plotG(self, X, y , predicted_X):
        plt.scatter(X, y)
        plt.plot(X, predicted_X, 'r--')
        plt.show()