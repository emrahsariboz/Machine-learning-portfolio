# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:21:45 2020

@author: EmrahSariboz
"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import accuracy_score
import numpy as np

data = load_iris()

X = data.data

y = data.target

updated_y = []

for i in range(len(y)):
    if y[i] == 0 :
        updated_y.append(1)
    elif y[i] == 1:
        updated_y.append(2)
    else:
        updated_y.append(3)
updated_y = np.array(updated_y)

row_dist = pd.DataFrame(squareform(pdist(X, metric='euclidean')))

row_clusters = linkage(pdist(X), method='average')

k = 3
clusters1 = fcluster(row_clusters, k, criterion='maxclust')

row_dendr = dendrogram(row_clusters)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()


print(clusters1)


def add_clusters_to_frame(or_data, clusters):
    or_frame = pd.DataFrame(data=or_data)
    or_frame_labelled = pd.concat([or_frame, pd.DataFrame(clusters)], axis=1)
    return(or_frame_labelled)

df = add_clusters_to_frame(X, clusters1)
df.columns = ['A', 'B', 'C', 'D', 'cluster']

predicted = df.cluster

print(accuracy_score(updated_y, df.cluster))