# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:12:45 2020

@author: EmrahSariboz
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("cali_renaw.csv")

#checking missing columns
missing_cols = [col for col in dataset.columns if dataset[col].isna().any() == True]

#Ratios column ratios
for i in missing_cols:
    print('The ratio of missing value at col ', i, ' is: ', dataset[i].isna().sum() / dataset.shape[0])
    
#Dropping the SOLAR because more than 65 percent of it is NaN
dataset =  dataset.drop('SOLAR', axis=1)   

#imputing the missing values
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

updated_dataset = pd.DataFrame(imp_mean.fit_transform(dataset.iloc[:, 1:]))

updated_dataset.insert(0, 'TIMESTAMP', dataset['TIMESTAMP'])

#Putting back the solumns that are lost after imputer
updated_dataset.columns = dataset.columns
updated_dataset.index = dataset.index
updated_dataset = updated_dataset.sort_values(['TIMESTAMP'])

#X values will be first 7 days. Thus slicing 168 hours
X = updated_dataset.iloc[0:2400, :]
y = updated_dataset.iloc[168:192, : ]

#Appending the sum of each day to the dataframe
first = 0
second = 24
for i in range(X.shape[0]):
    if i % 24 == 0:
        X = X.append(X.iloc[first:second, 1:].sum(axis = 0), ignore_index = True)
        second += 24
        first += 24


#Appending the 8th day to the dataframe
y = y.append(y.iloc[0:24, 1:].sum(axis = 0), ignore_index = True)        
        

#Creating days index for the TIMESTAMP dataset
X = pd.DataFrame(X.tail(100))
X = X.set_index(pd.Index([x for x in range(100)]))
X['TIMESTAMP'] = np.arange(1,101)
y = pd.DataFrame(y.tail(1))
y = y.set_index(pd.Index([1]))
y['TIMESTAMP'] = 8


#printing the correlation matrix
#It will produce the NaN values as 'hour', 'SOLAR THERMA' and 'SOLAR PV' are same for each day
X = X.drop(['Hour', 'SOLAR THERMAL'], axis = 1)
correlation_matrix = X.corr().round(2)
plt.figure(figsize = (16,8))
print(correlation_matrix)
sns.heatmap(data=correlation_matrix, annot=True, linewidths=.5)
plt.show()

#Since the BIOGAS and WINDTOTAL has the correlation, I will use this column 
#and 7 days to predict the total wind for the 8th day



X = X.loc[:, ['BIOMASS', 'SMALL HYDRO']]
#X = X.loc[:, ['GEOTHERMAL', 'WIND TOTAL']]
#y = y.loc[:, ['BIOGAS', 'WIND TOTAL']]

X = X.append(y)



X.to_csv('cali_renaw_cleaned-GEO-WIND.csv', index = False)