# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 05:27:51 2020

@author: EmrahSariboz
"""

import numpy as np

class Perceptron():
    
    def __init__(self, learningRate, n_iter = 50, rnd_state = 1):
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
    
        
        
    
