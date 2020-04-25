# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:21:45 2020

@author: EmrahSariboz
"""



from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import numpy as np
from sklearn import metrics


class my_aglomerative:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def getRowClusters(self, X, method):
        """
        Calculated the distance between each datapoint 
        and uses this to feed linkage library.
        """
        self.row_clusters = linkage(pdist(X), method = method)
        
        return self.row_clusters
    
    def getFcluster(self, rowCluster, k, criterion):
        return fcluster(rowCluster, k, criterion)
        
    def getDendogram(self, rowCluster):
        row_dendr = dendrogram(rowCluster)
        plt.title('Dendogram')
        plt.tight_layout()
        plt.ylabel('Distance')
        plt.xlabel('Clusters')
        plt.show()
    
    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        print(contingency_matrix)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix) 


    
