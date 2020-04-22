# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:24:42 2020

@author: EmrahSariboz
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

class my_ransac_model:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def fit(self, X_train, y_train):
        self.rg = RANSACRegressor()
        self.rg.fit(X_train, y_train)
    
    def predict(self, X_train, y_train):
        return self.rg.predict(X_train)
        
    def plotG(self, X, y , predicted_X):
        plt.scatter(X, y)
        plt.plot(X, predicted_X, 'r--')
        plt.show()