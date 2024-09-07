import numpy as np
from classes.learner.baselines.kernels import *

class KDE:
    def __init__(self, h, K=None, n_max=1000000):

        self.queries = np.zeros(n_max)
        self.evals = np.zeros(n_max)
        self.h = h
        if K == None:
            self.K = gaussian_kernel
        else:
            self.K  = K
        self.idx = 0


    def adjust_params(self, nu, norm_est, n, sigma):

        self.h = norm_est**(-2/(2*nu+1))*n**(-1/(2*nu+1))

    def get_data(self):
        par = {}
        par['Type'] = 'KDE'
        par['h'] = self.h
        par['kern'] = 'Gaussian'
        return par
        
    def query(self):
        self.queries[self.idx] = np.random.uniform(-1,1)
        return self.queries[self.idx]
    
    def update(self, y):
        self.evals[self.idx] = y
        self.idx += 1

    def train_model(self):
        pass

    def predict(self, x):
        num = den = 0
        for i in range(self.idx):
            num += self.evals[i]*self.K(x-self.queries[i], self.h)
            den += self.K(x-self.queries[i], self.h)
        return num/den