# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:45:46 2020

@author: EmrahSariboz
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("50_Startups.csv")

