# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:29:24 2020

@author: EmrahSariboz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from  sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:, 2:4]
y = dataset.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


#Fitting Logistic Regression to the TrainingSet

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(random_state=0)
lg.fit(X_train, y_train)

#Predicting the test set results
y_prediction = lg.predict(X_test)



#Making the Confussion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_prediction)

#Visualization of the training set results



from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(X_set[:,0].min() -1 , X_set[:, 0].max() + 1, step = 0.01),
                     np.arange (X_set[:, 1].min() - 1, X_set[:, 1].max() +1 , step = 0.01))



plt.xlim(X1.min(), X1.max())
plt.ylabel(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], 
                c = ListedColormap(('red', 'green'))(i), label =j)
    

plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()













