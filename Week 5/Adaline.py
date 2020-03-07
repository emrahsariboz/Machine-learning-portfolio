# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:33:55 2020

@author: EmrahSariboz
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

class AdalineGD():
    def __init__(self, learning_rate = 0.01, epoch = 50, random_state = 1):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1+X.shape[1])
        
        self.cost_= []
        
        for i in range(self.epoch):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            
            errors = y - output
            
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
        
    def net_input(self, X):
        print(np.dot(X, self.w_[1:]) + self.w_[0])
        print("fsdf")
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>= 0.0, 1, -1)



class Perceptron():
    def __init__(self, learning_rate, epoch= 50, random_state = 1):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1+X.shape[1])
        self.errors_ = []
        for i in range(self.epoch):
            error = 0
            self.prediction_result = [] 
            for xi, yi in zip(X, y):
                misclassified = (yi - self.predict(xi))
                self.prediction_result.append(int(self.predict(xi)))
                update = self.learning_rate * (misclassified)
                self.w_[1:] += update * xi
                self.w_[0]  += update
                error += int(misclassified != 0.0)
            print("Accuracy score on Epoch {} is {} ".format(i, accuracy_score(self.prediction_result, y)))
            print("The cost at epoch {} is {}".format(i, error))
            self.errors_.append(error)
        return -1
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        


ppn = AdalineGD(0.1, 10)

df = pd.read_csv("irishDataset.csv")


y= df.iloc[0:100, 4]
y = np.where(y == 'setosa', -1, 1)


X_irish = df.iloc[0:100, [0, 2]].values
ppn.fit(X_irish, y)



plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')

plt.show()

