# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:09:12 2020

@author: EmrahSariboz
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

def gradient_descent(x,y, learning_rate = 0.001, iteration = 1000):
    m_current = b_current = 0
    cost_list = []

    for i in range(iteration):
        y_predicted = m_current*x + b_current
        md = learning_rate * - (2 / len(x)) * sum( x * (y - y_predicted))
        mb = learning_rate * - (2 / len(x)) * sum( ( y - y_predicted))
        m_current = m_current - md
        b_current = b_current - mb
        cost_list.append(1 / (len(x)) * sum((y - y_predicted)))
        print('m {}, b {}, iteration {}'.format(m_current, b_current, i))
    
    return cost_list
    

cost_list = gradient_descent(X, y)



plt.plot(cost_list)
plt.ylabel('Error')
plt.show()