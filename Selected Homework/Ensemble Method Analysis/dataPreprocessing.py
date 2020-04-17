# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:37:54 2020

@author: EmrahSariboz
"""

import pandas as pd
from sklearn.impute import SimpleImputer

#Following line reads csv, adds column and converts the question marks to NaN values
df = pd.read_csv("mammographic_masses_converted.csv", names = ['BI-RADS', 'Age','Shape', 'Margin', 'Density', 'Severity'], na_values =["?"])
in_mean = SimpleImputer(strategy='median')



#Simple Imputer removes columns. Add them back!

X.columns = ['BI-RADS', 'Age','Shape', 'Margin', 'Density', 'Severity']


#Write the dataframe to CSV
X.to_csv("mammographic_masses_converted_cleaned.csv", header = True, index = False)