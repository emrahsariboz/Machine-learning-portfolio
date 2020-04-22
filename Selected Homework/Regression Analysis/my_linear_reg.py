# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:16:04 2020

@author: EmrahSariboz
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class my_linear_model:
    
    lm = LinearRegression()
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def fit(self, X, y):
        self.lm.fit(X, y)
    
    def predict(self, X, y):
        return self.lm.predict(X)
        
    def plotG(self, X, y , predicted_X):
        plt.scatter(X, y)
        plt.plot(X, predicted_X, 'r--')
        plt.show()