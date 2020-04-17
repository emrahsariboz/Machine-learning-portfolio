# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:27:21 2020

@author: EmrahSariboz
"""



from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class BaggingEns:
    classifierName = DecisionTreeClassifier(criterion = 'gini',max_depth=8, random_state = 1)
    #classifierName = SVC(gamma='auto', kernel='linear')
    bag = BaggingClassifier(base_estimator=classifierName, n_estimators=10)
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def fit(self, X, y):
        self.classifier = self.bag.fit(X, y)
        
    def predict(self, X_test):
        return self.classifier.predict(X_test)
    