# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 05:27:51 2020

@author: EmrahSariboz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    
    def __init__(self, learningRate = 0.01, n_iter = 50, rnd_state = 1):
        self.learningRate = learningRate
        self.n_iter = n_iter
        self.rnd_state = rnd_state
    
    def fit(self, X, y):
        
        '''
        X is training dataset
        y is labels
        '''
        
        rgen = np.random.RandomState(self.rnd_state)
        
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01,
                              size = 1 + X.shape[1])
        
        self.errors_  = []
        
        for i in range(self.n_iter):
            errors = 0
            
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:], self.w_[0])
    
    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 1, -1)



df = pd.read_csv("irishDataset.csv")

y = df.iloc[0:100, 4].values

y = np.where(y == 'setosa', -1, 1)

X= df.iloc[0:100, [0,2]].values

plt.scatter(X[:50, 0], X[:50, 1],
            color = 'red', marker='o', label='setosa')
        
    
plt.scatter(X[50:100, 0], X[50:100, 1],
            color = 'blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()