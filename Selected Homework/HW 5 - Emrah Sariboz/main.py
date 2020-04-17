# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:50:59 2020
@author: EmrahSariboz

"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import os
import sys
import timeit
import myKernelPCA
import myPCA
import myLDA
import seaborn as sns
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("classifier", help = "Name of the Classifier")
    parser.add_argument("component", help="Number of component")
    parser.add_argument("max_depth", help='depth of the tree')
    parser.add_argument("min_split", help='min node required for decision tree split ')
    parser.add_argument('--kernel', help = 'kernel for kernel_pca')
    parser.add_argument('--solver', help = 'solver for kernel_pca')
    parser.add_argument('--shrinkage', help = 'shrinkage for kernel_pca')
    parser.add_argument("datasetPassed", help = "Dataset Location")


    args = parser.parse_args()

    reduction_technique = str(args.classifier)
    
    component = int(args.component)
    
    m_depth = int(args.max_depth)
    
    m_split = int(args.min_split)
    
    kernel = str(args.kernel)
    
    solver = str(args.solver)
    
    shrinkage = str(args.shrinkage)
    
    dataset = str(args.datasetPassed)
    
    
    print('Reduction Technique: ', reduction_technique)
    
    print('Max depth of tree: ', m_depth)
    
    print('Min_samples_split of tree: ', m_split)

    print('Dataset: ', dataset)
    
    print('Number of component: ', component)
    

    #pass the dataset varible to if statement
    #example: if (dataset == dataset)
    isfile = os.path.isfile(dataset)
    
    #Check the existence of dataset and split it
    if isfile:
        df = pd.read_csv(dataset)
        X = df.iloc[:, 0:(df.shape[1]-1)]
        y = df.iloc[:, -1]
    else:
        if dataset == "mnist":
            mnist_dataset = datasets.fetch_openml('mnist_784')
            dataset_columns = np.array(['zero','one','two','three','four','five','six','seven','eight','nine'])
            X = mnist_dataset.data
            y = mnist_dataset.target
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify = y)
            
            X = X_test
            y = y_test
            
        else:
            print("\nThe provided dataset does not exist!")
            sys.exit()
            
    #If else statemets for each model
    if reduction_technique == "pca":
        
        start = timeit.default_timer()
      
        #object initialization
        pca =  myPCA.pca_reduction(X, y)
        
        
        #Normalize
        X = pca.normalize(X)            
    
        #Apply PCA
        X = pca.apply_PCA(X, component)

        sns.set()
        
        #Fit the decisionTreeAlgorithm
        print('Finished the PCA...\n')
        print('Shape of X after the PCA: ' , X.shape)
        print('Now applying cross-validation\n')
        
    
        pca.fit(X,y, m_depth, m_split)
        
        
        #print('Applyting grid search')
        #pca.grid_search(X,y)
        
        
        #end timer
        stop = timeit.default_timer()
        
        print('The total runtime of this classifier on ', dataset, ' dataset is: ', stop-start , ' seconds')
        
    elif reduction_technique == "kernel_pca":
        
        print('Kernel: ', kernel)
        
        start = timeit.default_timer()
      
        #object initialization
        kennel_pca =  myKernelPCA.kernel_pca_reduction(X, y)
        
            
        #Normalize
        X = kennel_pca.normalize(X)            
    
        #Apply PCA
        X = kennel_pca.apply_Kenel_PCA(X, component, kernel)

        sns.set()
        
        #Fit the decisionTreeAlgorithm
        print('Finished the Kernel PCA...\n')
        print('Shape of X after the Kernel PCA: ' , X.shape)
        print('Now applying cross-validation\n')
        
    
        kennel_pca.fit(X,y, m_depth, m_split)
        
        
        #print('Applyting grid search')
        #kennel_pca.grid_search(X,y)
        
        
        #end timer
        stop = timeit.default_timer()
        
        print('The total runtime of this classifier on ', dataset, ' dataset is: ', stop-start , ' seconds')
    
    elif reduction_technique == "lda":
        
        
        
        start = timeit.default_timer()
      
        #object initialization
        lda_analysis =  myLDA.linear_discriminant_analysis(X, y)
        
            
        #Normalize
        X = lda_analysis.normalize(X)            
    
        #Apply PCA
        X = lda_analysis.apply_LDA(X, y, solver, shrinkage)

        sns.set()
        
        #Fit the decisionTreeAlgorithm
        print('Finished the LDA...\n')
        print('Shape of X after the LDA: ' , X.shape)
        print('Now applying cross-validation\n')
        
    
        lda_analysis.fit(X,y, m_depth, m_split)
        
        
        #print('Applying grid search')
        #lda_analysis.grid_search(X,y)
        
        
        #end timer
        stop = timeit.default_timer()
        
        print('The total runtime of this classifier on ', dataset, ' dataset is: ', stop-start , ' seconds')
    else:
        print('Wrong parameter(s)')
        sys.exit()