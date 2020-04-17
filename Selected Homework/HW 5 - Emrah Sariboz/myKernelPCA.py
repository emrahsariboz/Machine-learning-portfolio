# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:32:37 2020
@author: EmrahSariboz

"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import KernelPCA

class kernel_pca_reduction:
    def __init__(self, X, y):
        self.X = X  
        self.y = y
        
    def normalize(self, X_train):
        model = StandardScaler()
        model.fit_transform(X_train)
        return model.fit_transform(X_train)
    
    def apply_Kenel_PCA(self, X_training, variance, kernel):
        pca = KernelPCA(variance, kernel = kernel, degree=4)
        pca.fit(X_training)       
        return pca.transform(X_training)
    
    def plot_data(self, X):
        plt.plot(X)
        plt.show()
    
    def fit(self, X, y, m_depth, m_split):
        scoring = {'accuracy': make_scorer(metrics.accuracy_score),
                   'precision': make_scorer(metrics.precision_score,  average = 'macro'),
                   'recall': make_scorer(metrics.recall_score, average = 'macro'),
                   'f1_macro': make_scorer(metrics.f1_score, average = 'macro')}
        
        cross_validate_result = cross_validate(DecisionTreeClassifier(max_depth=m_depth, min_samples_split=m_split ), X, y, cv=5, scoring=scoring, return_train_score=True)
        
        
        print('*'*60)
        print('Accuracy for each fold on testing dataset \n', cross_validate_result['test_accuracy'])
        print()
        print('Accuracy for each fold on training dataset \n', cross_validate_result['train_accuracy'])
        print()
        print('Mean of the accuracy for testing dataset  : ', cross_validate_result['test_accuracy'].mean())
        print()
        print('Mean of the accuracy for training dataset : ', cross_validate_result['train_accuracy'].mean())
        print()
        
        
        print('*'*60)
        
        
        print('Precision for each fold on testing \n', cross_validate_result['test_precision'])
        print()
        print('Precision for each fold on training dataset  \n', cross_validate_result['train_precision'])
        print()
        print('Mean of the precision for testing dataset: ', cross_validate_result['test_precision'].mean())
        print()
        print('Mean of the precision for training dataset: ', cross_validate_result['train_precision'].mean())
        print()
        
        
        print('*'*60)
        
        
        print('Recall for each fold on testing dataset \n', cross_validate_result['test_recall'])
        print()
        print('Recall for each fold on training dataset \n', cross_validate_result['train_recall'])
        print()
        print('Mean of the recall for testing dataset : ', cross_validate_result['test_recall'].mean())
        print()
        print('Mean of the recall for training dataset: ', cross_validate_result['train_recall'].mean())
        print()
        
        
        print('*'*60)
        
        
        print('F1 for each fold on testing dataset \n', cross_validate_result['test_f1_macro'])
        print()
        print('F1 for each fold on training dataset \n', cross_validate_result['train_f1_macro'])
        print()
        print('Mean of the f1 for testing dataset : ', cross_validate_result['test_f1_macro'].mean())
        print()
        print('Mean of the f1 for training dataset: ', cross_validate_result['train_f1_macro'].mean())
        print()
        
        print('*'*60)
        
    def grid_search(self, X, y):
        param_grid = {'max_depth':np.arange(3,10), 'min_samples_split':np.arange(2,7)}
        
        tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring='accuracy')
        
        tree.fit(X, y)
        
        print(tree.best_score_)
        print(tree.best_params_)