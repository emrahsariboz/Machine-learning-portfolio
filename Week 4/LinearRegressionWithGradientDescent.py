import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd


df = pd.read_csv("C:/Users/Emrah Sariboz/GitHubProjects/Machine-learning-portfolio/Week 5/irishDataset.csv")



X = df.iloc[0:100, [0,2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == "setosa", -1, 1)
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


class AdalineSGD():
    
    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    
    def fit(self, X, y):
        
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        
        self.cost_ = []
        
        for i in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            cost = []
            
            for xi, yi in zip(X, y):
                cost.append(self._update_weights(xi, yi))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _update_weights(self, xi, yi):
        
        output = self.net_input(xi)
        
        error = (yi - output)
        cost = (1/2) * (error **2)
        
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
     
        return cost
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        if self.net_input(X) >= 0.0 :
            return 1
        return -1

ada = AdalineSGD(n_iter = 15, eta= 0.01, random_state=1)
ada.fit(X_std, y)
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel("Epoch")
plt.ylabel("Average Cost")
plt.show()