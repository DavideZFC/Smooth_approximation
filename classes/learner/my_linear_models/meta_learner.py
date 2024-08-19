import numpy as np
from classes.learner.my_linear_models.bases import *
from sklearn.linear_model import LinearRegression
from classes.learner.my_linear_models.optimal_design import *
import random


class MetaLearner:
    def __init__(self, basis, n_arms, d, optimal_deisgn=False, n_max=1000000, new_idea=False):
        # dimension of the problem
        self.d = d
        self.basis = basis
        self.optimal_design = optimal_deisgn
        self.n_max = n_max
        self.newidea = new_idea

        # arms avaiable
        self.n_arms = n_arms
        self.arms = np.linspace(-1,1,n_arms)
        self.queries = np.zeros(n_max, dtype=np.int16)
        self.evals = np.zeros(n_max)
        self.idx = 0

        self.linarms = self.apply_feature_map(self.arms)
        if optimal_deisgn:
            self.compute_optimal_design()
            if self.newidea:
                self.compute_fixed_design()
        print('newidea = {}, nmax = {}, optimal_design = {}'.format(self.newidea, self.n_max, self.optimal_design))

    def get_data(self):
        par = {}
        par['Type'] = 'linReg'
        par['d'] = self.d
        par['basis'] = self.basis
        par['optimal'] = self.optimal_design
        return par

    def apply_feature_map(self, x):
  
        n = len(x)
        if self.basis == 'Fourier':
            X = make_sincos_arms(n, self.d, x)
        elif self.basis == 'Legendre':
            X = make_legendre_arms(n, self.d, x)
        elif self.basis == 'Chebishev':
            X = make_chebishev_arms(n, self.d, x)
        elif self.basis == 'Legendre_norm':
            X = make_legendre_norm_arms(n, self.d, x)
        elif self.basis == 'Poly':
            X = make_poly_arms(n, self.d, x)
        else:
            raise Exception("Sorry, basis not found")
        
        return X

    def query(self):
        if self.optimal_design:
            return self.query_optimal_design()
        else:
            return self.query_uniform()
        
    def compute_optimal_design(self):
        self.pi = find_optimal_design(self.linarms)

    def compute_fixed_design(self):
        times_to_pull = np.ceil(self.pi*self.n_max).astype(np.int16)
        total = int(np.sum(times_to_pull))
        ideal_queries = np.zeros(total, dtype=np.int16)
        idx = 0
        for i in range(len(times_to_pull)):
            if times_to_pull[i] > 0:
                ideal_queries[idx:idx+times_to_pull[i]] = i
                idx += times_to_pull[i]
        random.shuffle(ideal_queries)
        self.queries[:self.n_max] = ideal_queries[:self.n_max]

        
    def query_optimal_design(self):
        if self.newidea:
            pass
        else:
            self.queries[self.idx] = np.random.choice(np.arange(self.n_arms), p=self.pi)
        return self.arms[self.queries[self.idx]]

    def query_uniform(self):
        self.queries[self.idx] = np.random.randint(self.n_arms)
        return self.arms[self.queries[self.idx]]
    
    def update(self, y):
        self.evals[self.idx] = y
        self.idx += 1

    def train_model(self):
        X = self.linarms[self.queries[:self.idx]]
        y = self.evals[:self.idx]

        self.predictior = LinearRegression()
        self.predictior.fit(X,y)

    def print_coef(self):
        return self.predictior.coef_

    def predict(self, x):
        self.train_model()
        X = self.apply_feature_map(x)
        y = self.predictior.predict(X)
        return y


    


