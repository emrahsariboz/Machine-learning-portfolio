# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:20:50 2020

@author: EmrahSariboz
"""
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn import metrics

class dbscan_class:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def fit_predict(self, X, epsilon, min_sample):
        self.dbs = DBSCAN(eps=epsilon, min_samples=min_sample)
        return self.dbs.fit_predict(X)
    
    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(contingency_matrix)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
    
    def optimal(self, X, y_true):
        prediction_score = {}
        purest_cluster = {}
        
        eps_range = np.arange(0.1, 1.1, 0.1)
        
        print(eps_range)
        
        for i in eps_range:
            self.dbs = DBSCAN(eps=i)
            prediction_score[i] = self.purity_score(y_true, self.dbs.fit_predict(X))
            
        max_eps_key = max(prediction_score, key=prediction_score.get)
        #max_eps_value = max(prediction_score.iteritems(), key=operator.itemgetter(1))[1]
        
        print('Finding best eps value')
        
        for i in range(5, 10):
            self.dbs = DBSCAN(eps=max_eps_key, min_samples = i)
            purest_cluster[i] = self.purity_score(y_true, self.dbs.fit_predict(X))
        
        
        min_sample_key = max(purest_cluster, key=purest_cluster.get)
        #max_sample_value = max(purest_cluster.iteritems(), key=operator.itemgetter(1))[1]
            
    
        print('The optimal epsilon and the min sample value is', max_eps_key , ' ' , min_sample_key)
        self.dbs = DBSCAN(eps=max_eps_key, min_samples=min_sample_key)
        print('Using these values we get following cluster purity' 
            , self.purity_score(y_true, self.dbs.fit_predict(X)))