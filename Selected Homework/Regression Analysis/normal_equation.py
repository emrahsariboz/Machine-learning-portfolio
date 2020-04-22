# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:53:36 2020

@author: EmrahSariboz
"""
import numpy as np
import matplotlib.pyplot as plt

class normalE:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def get_coefficients(self, X, y):
        ones_constructed = np.hstack((np.ones((X.shape[0], 1)), X))
        params = np.linalg.inv(np.dot(ones_constructed.T, ones_constructed))
        return np.dot(params, np.dot(ones_constructed.T, y))
    
    def predict(self, X, params):
        return params[0] + np.dot(params[1], X.T) 
    
    def plotG(self, X, y , predicted_X):
        plt.scatter(X, y)
        plt.plot(X, predicted_X, 'r--')
        plt.show()