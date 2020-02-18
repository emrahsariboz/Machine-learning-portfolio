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
X_train = sc_X.fit(X_train)
X_test = sc_X.fit(X_test)



