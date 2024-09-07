import numpy as np
from classes.learner.baselines.kernels import *

def fatt(i):
    if i == 0:
        return 1
    return i*fatt(i-1)

class NW:
    def __init__(self, h, K=None, n_max=10000, nu=5):

        self.queries = np.zeros(n_max)
        self.evals = np.zeros(n_max)
        self.nu = nu
        self.h = h
        if K == None:
            self.K = gaussian_kernel
        else:
            self.K  = K
        self.idx = 0
    
    def adjust_params(self, nu, norm_est, n, sigma):
        self.nu = nu
        self.h = norm_est**(-2/(2*nu+1))*n**(-1/(2*nu+1))

    def get_data(self):
        par = {}
        par['Type'] = 'NW'
        par['h'] = self.h
        par['kern'] = 'Gaussian'
        return par
        
    def query(self):
        self.queries[self.idx] = np.random.uniform(-1,1)
        return self.queries[self.idx]
    
    def update(self, y):
        self.evals[self.idx] = y
        self.idx += 1

    def phi(self, x):
        vec = np.zeros((self.nu,1))
        for i in range(self.nu):
            vec[i,0] = x**i/fatt(i)
        return vec

    def train_model(self):
        pass

    def predict(self, X):
        return_vec = np.zeros_like(X)
        n = len(X)

        for j in range(n):
            x = X[j]

            # build matrix B_nx and vector a_nx

            Bnx = np.zeros((self.nu, self.nu))
            anx = np.zeros((self.nu, 1))

            for i in range(self.idx):
                delta = (x-self.queries[i]) / self.h
                U = self.phi(delta)
                Bnx += np.dot(U,U.T)*self.K(delta,1)
                anx += U*self.evals[i]*self.K(delta,1)

            return_vec[j] = np.linalg.solve(Bnx, anx)[0]
        
        return return_vec

