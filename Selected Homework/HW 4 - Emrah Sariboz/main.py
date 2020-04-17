# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:50:59 2020
@author: EmrahSariboz

"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import pandas as pd
import AdaBoostC
import os
import sys
import timeit
import RandomFC
import BaggingC
from sklearn.svm import SVC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("classifier", help = "Name of the Classifier")
    parser.add_argument("datasetPassed", help = "Dataset Location")
    args = parser.parse_args()

    classifierName = str(args.classifier)
    dataset = str(args.datasetPassed)
    
    #pass the dataset varible to if statement
    #example: if (dataset == dataset)
    isfile = os.path.isfile(dataset)
    
    #Check the existence of dataset and split it
    if isfile:
        df = pd.read_csv(dataset)
        X = df.iloc[:, 0:(df.shape[1]-1)]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    else:
        if dataset == "digit":
            #Loading the digits dataset
            digist_dataset = datasets.load_digits()
            X = digist_dataset.data
            y = digist_dataset.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        else:
            print("\nThe provided dataset does not exist!")
            sys.exit()
            
    #Comparison
    #comparison = SVC(kernel = 'rbf')
    #comparison.fit(X_train, y_train)
    #print(accuracy_score(y_test, comparison.predict(X_test)))

    #If else statemets for each model
    if classifierName == "randomForest":
        start = timeit.default_timer()
        classifier =  RandomFC.RandomForest(X_train, y_train)
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        prediction_training = classifier.predict(X_train);
        stop = timeit.default_timer()
        print('The total runtime of this classifier on ', dataset, ' dataset is: ', stop-start , ' seconds')
        print('\nThe accuracy of random forest classifier on test dataset is: ', accuracy_score(y_test, prediction))
        print('\nThe accuracy of random forest classifier on training dataset is: ', accuracy_score(y_train, prediction_training))
    elif classifierName == "bagging":
        start = timeit.default_timer()
        classifier =  BaggingC.BaggingEns(X_train, y_train)
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        prediction_training = classifier.predict(X_train);
        stop = timeit.default_timer()
        print('The total runtime of this classifier is: ', stop-start , ' seconds')
        print('\nThe accuracy of bagging classifier on test dataset is: ', accuracy_score(y_test, prediction))
        print('\nThe accuracy of bagging classifier on training dataset is: ', accuracy_score(y_train, prediction_training))
    elif classifierName == "adaboost":
        start = timeit.default_timer()
        classifier =  AdaBoostC.AdaBoostEns(X_train, y_train)
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        prediction_training = classifier.predict(X_train);
        stop = timeit.default_timer()
        print('The total runtime of this classifier is: ', stop-start , ' seconds')
        print('\nThe accuracy of adaboost classifier on test dataset is: ', accuracy_score(y_test, prediction))
        print('\nThe accuracy of adaboost classifier on training dataset is: ', accuracy_score(y_train, prediction_training))
    else:
        print('Wrong parameter(s)')