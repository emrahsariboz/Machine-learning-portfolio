# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:37:07 2020

@author: EmrahSariboz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Social_Network_Ads.csv")

X = df.iloc[:, [2, 3]].values
y = df.iloc[:,  4].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##Classifier
from sklearn.svm import SVC
classifier = SVC(kernel= 'linear', random_state = 0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(X_set[:,0].min() -1 , X_set[:, 0].max() + 1, step = 0.01),
                     np.arange (X_set[:, 1].min() - 1, X_set[:, 1].max() +1 , step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
        alpha = 0.75, cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], 
                c = ListedColormap(('red', 'green'))(i), label =j)
    

plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()





