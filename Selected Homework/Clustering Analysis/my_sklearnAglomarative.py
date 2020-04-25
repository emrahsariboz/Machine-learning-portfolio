# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:42:46 2020

@author: EmrahSariboz
"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


class my_sklearn_aglo:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def fit_predict(self, X, n_cluster, linkage):
        agc = AgglomerativeClustering(n_clusters= n_cluster, linkage = linkage)
        return agc.fit_predict(X)
    
    def purity_score(self, y_true, predicted_y):
        print(y_true.shape)
        print(predicted_y.shape)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, predicted_y)
        return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix) 
    
