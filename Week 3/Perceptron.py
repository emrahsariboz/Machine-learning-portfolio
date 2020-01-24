# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 03:45:01 2020

@author: EmrahSariboz
"""

import random


class Percepton:
    weights = []
    def __init__(self, features):
        self.features = features
        
        #initialize weights randomly
        for i in range(len(features)):
            self.weights.append(random.randrange(-1, 1))
    
    def sign(self, n):
        if n >= 0:
            return 1
        return -1
        
    def guess(self, inputs):
        sum_of_weights = 0
        for i in range(len(inputs)):
            sum_of_weights += self.weights[i] * inputs[i]
        
        return self.sign(sum_of_weights)
        
    

c


inputs = [-1, 0.5]
p = Percepton(inputs)
print(p.guess(inputs))
