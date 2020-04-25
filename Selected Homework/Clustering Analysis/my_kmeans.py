# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:51:36 2020

@author: EmrahSariboz
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
class my_kmeans:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
               
    def fit(self, X, n_cluster):
        
        self.km = KMeans(n_clusters=n_cluster)
        self.km.fit(X)


    def predict(self, X):
        
        prediction = self.km.predict(X)
        
        return prediction
    
    def elbow_method(self, X):
        
        self.distertion = []
        self.values = [x for x in range(1, 10)]
        for i in range(1, len(self.values)+1):
            self.km = KMeans(n_clusters=i)
            self.km.fit(X)


            self.distertion.append(self.km.inertia_)
        
        
        
        plt.title('Elbow Method')
        plt.xlabel('K')
        plt.ylabel('Objective Function')
        plt.plot(self.values, self.distertion)           
        plt.show()
        
    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        print(contingency_matrix)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
    
    
    
    
    
    
    
