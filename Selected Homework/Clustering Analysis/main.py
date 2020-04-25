# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:50:59 2020
@author: EmrahSariboz

"""

from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import os
import timeit 
import sys
import seaborn as sns
import numpy as np
from sklearn import datasets
import my_kmeans
import my_aglomerative
import my_sklearnAglomarative
import my_dbscan

if __name__ == "__main__":
    

    sns.set()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("clustering_algo", help = "Name of the clustering algorithm")
   
    parser.add_argument("datasetPassed", help = "Dataset Location")

    parser.add_argument('--n_cluster', help = 'totalNumberOfCluter')
    
    parser.add_argument('--criterion', help = 'criterion')
    
    parser.add_argument('--epsilon', help = 'epsilon')
    
    parser.add_argument('--min_sample', help = 'min_sample')
    
    
    args = parser.parse_args()
    
    clustering_algo = str(args.clustering_algo)
    
    if args.n_cluster == None:
        n_cluster = (args.n_cluster)
    else:
        n_cluster = int(args.n_cluster)
    
    
    dataset = (args.datasetPassed)
    
    if args.criterion == None:
        criterion = 'average' # Setting the default criterion for those commands that has no criterion specification.
    else:
        criterion = str(args.criterion)
    
    
    
    
    isfile = os.path.isfile(dataset)
    
    #Check the existence of dataset and split it
    if isfile:
        df = pd.read_csv(dataset, index_col = False)
        
        X = pd.DataFrame(df.iloc[:, 0:(df.shape[1]-1)])
                
        y = pd.DataFrame(df.iloc[:, -1])

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
    elif dataset == "mnist":
            mnist_dataset = datasets.fetch_openml('mnist_784')
            dataset_columns = np.array(['zero','one','two','three','four','five','six','seven','eight','nine'])
            X = mnist_dataset.data
            y = mnist_dataset.target
            
            #Subset of the mnist dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify = y)
            
            X = X_test
            y = y_test
    
    else:
        print("\nThe provided dataset does not exist!")
        sys.exit()
        
    if clustering_algo == 'k_means':
        
        #time-start
        start = timeit.default_timer()
        
        #object initialization
        km = my_kmeans.my_kmeans(X, y)
        
        if n_cluster == None:
            n_cluster = 3
        
        #fit
        km.fit(X, n_cluster)
        
        #predict
        predicted_y = km.predict(X)
        
        #elbow method
        km.elbow_method(X)    
    

                
        #time-stop
        stop = timeit.default_timer()
        
        #Cluster Quality
        print('Purity of the training cluster ', km.purity_score(y, predicted_y)) 
        
        #Total Time
        print('Total time taken ', (stop - start))
    
    elif clustering_algo == 'scipy_agg':
        #time-start
        start = timeit.default_timer()
    
        #object initiazation
        scipy_agg = my_aglomerative.my_aglomerative(X, y)
        
        #Get row clusteris
        row_cluster = scipy_agg.getRowClusters(X, criterion)
        
        if n_cluster == None:
            n_cluster = 3
        
        #Get labels
        predicted_y = scipy_agg.getFcluster(row_cluster, k = n_cluster, criterion='maxclust')
        
        #Print Dendogram
        scipy_agg.getDendogram(row_cluster)
        
        #time-stop
        stop = timeit.default_timer()
        
        #cluster purity
        print('Purity of the training cluster ', scipy_agg.purity_score(y, predicted_y)) 
    
        #Total Time
        print('Total time taken ', (stop - start))
    elif clustering_algo == 'sklearn_agg':
        
        #time-start
        start = timeit.default_timer()
        
        #object initialization
        sklearn_agg = my_sklearnAglomarative.my_sklearn_aglo(X, y)
        
        
        if n_cluster == None:
            n_cluster = 3
        
        if criterion == None:
            criterion = 'complete'
        
        #fit_predict
        predicted_y = sklearn_agg.fit_predict(X, n_cluster, criterion)
        print(predicted_y)
        
        #time-stop
        stop = timeit.default_timer()
        
        #cluster purity
        print('Purity of the training cluster ', sklearn_agg.purity_score(y, predicted_y)) 

        #Total Time
        print('Total time taken ', (stop - start))
    elif clustering_algo == 'dbscan':
        

        
        if args.epsilon == None:
            epsilon = (args.epsilon)
            epsilon = 0.5
        else:
            epsilon = float(args.epsilon)
            
        
        if args.min_sample == None:
            min_samples = (args.min_sample)
            min_samples = 5
        else:
            min_samples = int(args.min_sample)


        
        #time-start
        start = timeit.default_timer()
        
        #object initialization
        sklearn_dbscan = my_dbscan.dbscan_class(X, y)

        #predict
        predicted_y = sklearn_dbscan.fit_predict(X, epsilon, min_samples)
        
        #cluster purity
        print('Purity of the training cluster ', sklearn_dbscan.purity_score(y, predicted_y))
        
        print('Finding the best epsilon and min_sample values')
        
        #time-stop
        stop = timeit.default_timer()
        
        #Total Time
        print('Total time taken ', (stop - start))
        
        #finding optimal epsilon
        sklearn_dbscan.optimal(X, y)
        
        
        
    else:
        print('Wrong parameter(s)')
        sys.exit()