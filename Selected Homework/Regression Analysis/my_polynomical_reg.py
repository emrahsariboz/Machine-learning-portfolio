# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:57:12 2020

@author: EmrahSariboz
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
class my_poly_linear:
    

    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def fit_polynomial(self, X, degree):
        
        #print('Degree', degree)
        
        self.polyFeatures = PolynomialFeatures(degree=degree)
    
        self.lrd = LinearRegression()
        
        return self.polyFeatures.fit_transform(X)
        
    def fit(self, X, y):
        self.lrd.fit(X,y)
    
    def predict(self, X, y):
        return self.lrd.predict(X)
        
    def plotG(self, X, y , predicted_X):
        plt.scatter(X, y)
        plt.plot(X, predicted_X, 'r--')
        plt.show()