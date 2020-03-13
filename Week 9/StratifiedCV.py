# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 01:52:19 2020

@author: EmrahSariboz
"""

from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

#The ration of 0 and 1 in y is approximately 1/10
X, y = np.ones((50,1)), np.hstack(([0] * 45, [1] * 5))

skf = StratifiedKFold(n_splits=3)

for train, test in skf.split(X,y):
    print('train - {} | test - {}'.format(np.bincount(y[train]), np.bincount(y[test])))


kf = KFold(n_splits=3)
for train, test in kf.split(X,y):
    print('train - {} | test - {}'.format(np.bincount(y[train]), np.bincount(y[test])))
