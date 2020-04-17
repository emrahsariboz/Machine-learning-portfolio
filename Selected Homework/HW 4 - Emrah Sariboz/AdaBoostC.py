# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:27:21 2020

@author: EmrahSariboz
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

class AdaBoostEns:
    classifierName = DecisionTreeClassifier(criterion = 'gini', max_depth=10, random_state = 1)
    #classifierName = SVC(probability=True, kernel='linear')
    boost = AdaBoostClassifier(base_estimator=classifierName, 
                               n_estimators=50, learning_rate = 1, random_state= 1)
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def fit(self, X, y):
        self.classifier = self.boost.fit(X, y)
        
    def predict(self, X_test):
        return self.classifier.predict(X_test)




# =============================================================================
# weights = np.array([0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.167, 0.167, 0.167, 0.072])
# y = np.array([1,1,1,-1,-1,-1,1,1,1,-1])
# y_hat= np.array([1,1,1,-1,-1,-1,1,-1,-1,-1])
# 
# equality = []
# 
# for i in range(10):
#     if y[i] == y_hat[i]:
#         equality.append(0)
#     else:
#         equality.append(1)
#         
# error_rate = np.dot(weights , np.array(equality))
# 
# 
# alpha_t = 0.5 * np.log((1 - error_rate)/error_rate)
# 
# print('Alpha is ', alpha_t)
# 
# for i in range(0, len(equality)):
#     #print('Index ',i,', 0.072 * exp(-0.35 x ', y[i] , ' x ', y_hat[i], ') 
#     print('Index {} {} '.format(i, weights[i] * np.exp(-0.35 * y[i] * y_hat[i])))
# 
# weights2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# 
# y = np.array([1,1,1,-1,-1,-1,1,1,1,-1])
# y_hat= np.array([1,1,1,-1,-1,-1,-1,-1,-1,-1])
# 
# 
# equality2 = []
# 
# for i in range(10):
#     if y[i] == y_hat[i]:
#         equality2.append(0)
#     else:
#         equality2.append(1)
#         
# error_rate = np.dot(weights2 , np.array(equality2))
# 
# 
# #alpha_t = 0.5 * np.log((1 - error_rate)/error_rate)
# 
# alpha_t = 0.424
# 
# print('Alpha is ', alpha_t)
# 
# print()
# 
# for i in range(0, len(equality2)):
#     #print('Index ',i,', 0.072 * exp(-0.35 x ', y[i] , ' x ', y_hat[i], ') 
#     print('Index {} {} '.format(i, weights2[i] * np.exp(-0.424 * y[i] * y_hat[i])))
# =============================================================================
