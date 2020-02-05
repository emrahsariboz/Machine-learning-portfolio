# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:33:55 2020

@author: EmrahSariboz
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class AdalineGD():
    
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1+X.shape[1])
        
        self.cost_= []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            
            errors = y - output
            
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def activation(self, X):
        return X
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>= 0.0, 1, -1)




class Perceptron():
    def __init__(self, learning_rate, num_iter= 50, random_state = 1):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1+X.shape[1])
        self.errors_ = []
        
        for i in range(self.num_iter):
            errors = 0
            
            for xi, target in zip(X, y):
                update = self.learning_rate * (target -self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)



ppn = Perceptron(learning_rate=0.1, num_iter=10)

df = pd.read_csv("irishDataset.csv")


y= df.iloc[0:100, 4]
y = np.where(y == 'setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values


# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values



fix, ax = plt.subplots(nrows=1, ncols=2, figsize = (10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.0001).fit(X,y)

ax[0].plot(range(1, len(ada1.cost_) + 1),
  np.log10(ada1.cost_), marker = 'o')


