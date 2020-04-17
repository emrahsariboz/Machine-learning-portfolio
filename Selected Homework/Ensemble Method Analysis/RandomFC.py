# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:02:19 2020

@author: EmrahSariboz
"""

from sklearn.ensemble import RandomForestClassifier
class RandomForest:
    classifierName = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 6, random_state = 1)
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def fit(self, X, y):
        self.classifier = self.classifierName.fit(X, y)
        
    def predict(self, X_test):
        return self.classifier.predict(X_test)
    