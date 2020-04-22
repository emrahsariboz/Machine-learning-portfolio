# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:50:59 2020
@author: EmrahSariboz

"""

from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import timeit 
import sys
import normal_equation
import seaborn as sns
import numpy as np
import my_linear_reg
import my_rid_reg
import my_lasso_reg
import my_polynomical_reg
import matplotlib.pyplot as plt
import my_ransac_reg
import uuid
from sklearn.preprocessing import StandardScaler
if __name__ == "__main__":
    
    DEFAULT = uuid.uuid4()

    sns.set()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("regressor", help = "Name of the regressor")
   
    parser.add_argument('--alpha', help = 'alpha')
    
    parser.add_argument('--solver', help = 'solver')
    
    parser.add_argument('--degree', help = 'degree')

    
    parser.add_argument('--shrinkage', help = 'shrinkage for kernel_pca')

    parser.add_argument("datasetPassed", help = "Dataset Location")

    args = parser.parse_args()
    
    regression_technique = str(args.regressor)
    
    alpha = (args.alpha)
    
    solver = (args.solver)
    
    degree = (args.degree)
    
    dataset = (args.datasetPassed)
    
   

    #pass the dataset varible to if statement
    #example: if (dataset == dataset)
    isfile = os.path.isfile(dataset)
    
    #Check the existence of dataset and split it
    if isfile:
        df = pd.read_csv(dataset, index_col = False)
        
        if dataset == 'BostonHousing.csv':
            
            
            X = df['lstat']
            y = df['medv']
           
            X = np.array(X).reshape(-1, 1)  
            y = np.array(y).reshape(-1, 1)
                       
#            plt.title('Boston Housing')
#            plt.scatter(X,y)
#            plt.xlabel('lstat')
#            plt.ylabel('medv')
#            plt.show()
            
#            plt.xlabel('Average number of rooms')
#            plt.ylabel('Price')
            
        else:
            X = pd.DataFrame(df.iloc[:, 0:(df.shape[1]-1)])
            y = pd.DataFrame(df.iloc[:, -1])
#            plt.title('Cali Renawed')
#            plt.scatter(X,y)
#            plt.xlabel('BIOMASS')
#            plt.ylabel('SMALL HYDRO')
#            plt.show()
            
            
#            plt.xlabel('BIOMASS')
#            plt.ylabel('SMALL HYDRO')
#        ss = StandardScaler()
#        
#        ss.fit(X)
#        y = ss.transform(y)
#        X = ss.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#        plt.title(dataset)     
        
        
    else:
        print("\nThe provided dataset does not exist!")
        sys.exit()
        
    if regression_technique == 'normal_reg':
        
        start = timeit.default_timer()
        
        #I analysted and found out that 'LSTAT' and 'MEDV' column has strong negative correlation.
        #Thus, I will only use that column
                        
        #object initialization
        nor_eq = normal_equation.normalE(X_train, y_train)
        
        #Getting coefficients
        params = nor_eq.get_coefficients(X_train, y_train)
        
        #predict on X_train
        predictred_X_train = nor_eq.predict(X_train, params)
        
        #Predict on X_test
        predicted_X = nor_eq.predict(X_test, params)
        
        #stop
        stop = timeit.default_timer()

        #Mean Squared Errors
        print('MSE for normal equation on training dataset:  ' , mean_squared_error(y_train, predictred_X_train))

        #Mean Squared Errors
        print('MSE for normal equation on testing dataset:  ' , mean_squared_error(y_test, predicted_X))

        #Total Time
        print('The total runtime of the linear regression on ', dataset, ' dataset is: ', stop-start , ' seconds')

        #plot
        print('Plotting the regression line')
        
#        nor_eq.plotG(X_test, y_test, predicted_X)

    
    elif regression_technique == "linear_reg":
        
        
        start = timeit.default_timer()
      
        #object initialization
        reg = my_linear_reg.my_linear_model(X_train, y_train)
        
        #fit
        reg.fit(X_train, y_train) 
        
        #predict
        x_train_pred = reg.predict(X_train, y_train)
        
        #predict
        x_test_pred = reg.predict(X_test, y_test)

        #stop
        stop = timeit.default_timer()
        
        #Mean Squared Errors
        print('MSE for linear regression on training dataset: ' , mean_squared_error(y_train, x_train_pred))
        
        
        #Mean Squared Errors
        print('MSE for linear regression on testing datset: ' , mean_squared_error(y_test, x_test_pred))
              
        
        #Total Time
        print('The total runtime of the linear regression on ', dataset, ' dataset is: ', stop-start , ' seconds')

        #Plot
#        plt.title('Testing')
#        reg.plotG(X_test, y_test, x_test_pred)

        
        #Plot
#        plt.title('Training')
#        reg.plotG(X_train, y_train, x_train_pred)

    elif regression_technique == "ridge_reg":
        
        start = timeit.default_timer()
      
        #object initialization
        reg = my_rid_reg.my_ridge_model(X_train, y_train)
        
               
        if alpha == None:
            alpha = 2
        
        if solver == None:
            solver = 'auto'
        
        
        print('Alpha and solver ' , alpha, solver)
        
        #fit
        reg.fit( X_train, y_train, alpha, solver)         
        
        #predict 
        x_trainn_predict = reg.predict(X_train, y_train)
        
        #predict
        x_test_predict = reg.predict(X_test, y_test)

        #stop
        stop = timeit.default_timer()
        
        #Mean Squared Errors
        print('MSE for ridge regression on training dataset: ' , mean_squared_error(y_train, x_trainn_predict))

        #Mean Squared Errors
        print('MSE for ridge regression on test dataset : ' , mean_squared_error(y_test, x_test_predict))
        
        #Total Time
        print('The total runtime of the ridge regression on ', dataset, ' dataset is: ', stop-start , ' seconds')
        
        #Plot
#        reg.plotG(X_test, y_test, x_test_predict)
        
        
    elif regression_technique == "lasso_reg":
        
        
        start = timeit.default_timer()
      
        #object initialization
        reg = my_lasso_reg.my_lasso_model(X_train, y_train)
        
        if degree == None:
            degree = 2
        
        #fit
        reg.fit( X_train, y_train,int(degree))         
        
        #predict 
        x_train_predict = reg.predict(X_train, y_train)
        
        #predict
        x_test_predict = reg.predict(X_test, y_test)
        
        #Mean Squared Errors
        print('MSE for lasso reqression on training dataset: ' , mean_squared_error(y_train, x_train_predict))

        #stop
        stop = timeit.default_timer()

        #Mean Squared Errors
        print('MSE for lasso reqression on test dataset : ' , mean_squared_error(y_test, x_test_predict))
        
        
        #Total Time
        print('The total runtime of the lasso regression on ', dataset, ' dataset is: ', stop-start , ' seconds')

        #Plot
#        reg.plotG(X_test, y_test, x_test_predict)

        
    elif regression_technique == "poly_lr":
        
        start = timeit.default_timer()
      
        #object initialization
        reg = my_polynomical_reg.my_poly_linear(X_train, y_train)
        
        #fit Polynomical Features
        
        print("Degree before fit", degree)
        
        if degree == None:
            degree = 2
        X = reg.fit_polynomial(X, int(degree))         
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        #fit
        reg.fit(X_train, y_train)
        
        #predict 
        x_train_predict = reg.predict(X_train, y_train)
        
        #predict
        x_test_predict = reg.predict(X_test, y_test)

        #stop
        stop = timeit.default_timer()

        #Mean Squared Errors
        print('MSE for polynomial featured linear regression on training dataset: ' , mean_squared_error(y_train, x_train_predict))

        #Mean Squared Errors
        print('MSE for polynomial featured linear regression on test dataset : ' , mean_squared_error(y_test, x_test_predict))

        #Total Time
        print('The total runtime of the polynomial featured linear regression on ', dataset, ' dataset is: ', stop-start , ' seconds')

    elif regression_technique == "ransac_reg":
        
        start = timeit.default_timer()
      
        #object initialization
        reg = my_ransac_reg.my_ransac_model(X_train, y_train)
        
        #fit
        reg.fit( X_train, y_train)         
        
        #predict 
        x_train_predict = reg.predict(X_train, y_train)
        
        #predict
        x_test_predict = reg.predict(X_test, y_test)
        
        #stop
        stop = timeit.default_timer()

        #Mean Squared Errors
        print('MSE for ransac regression on training dataset: ' , mean_squared_error(y_train, x_train_predict))

        #Mean Squared Errors
        print('MSE for ransac regression on test dataset : ' , mean_squared_error(y_test, x_test_predict))
        
        #Total Time
        print('The total runtime of the ransac regression on ', dataset, ' dataset is: ', stop-start , ' seconds')
        
        #Plot
#        reg.plotG(X_test, y_test, x_test_predict)
    else:
        print('Wrong parameter(s)')
        sys.exit()