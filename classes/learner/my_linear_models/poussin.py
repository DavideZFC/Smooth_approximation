import numpy as np
from classes.learner.my_linear_models.meta_learner import MetaLearner


class Poussin(MetaLearner):

    def __init__(self, basis, n_arms, d, optimal_deisgn=False, n_max=10000, new_idea=False, n_pous=5):
        super().__init__(basis, n_arms, d, optimal_deisgn, n_max, new_idea)
        self.n_pous = n_pous
        self.sample_all_noises()

    def get_data(self):
        par = {}
        par['Type'] = 'Poissin'
        par['d'] = self.d
        par['n_pous'] = self.n_pous
        par['optimal'] = self.optimal_design
        return par


    def query(self):
        if self.optimal_design:
            x_prenoise = self.query_optimal_design()
        else:
            x_prenoise = self.query_uniform()
        return self.adjust(x_prenoise+self.noises[self.idx])
        

    def sample_all_noises(self):

        def PoussinWrapper(n,p):
            c1 = (2*n+1-p)/2
            c2 = (p+1)/2
            c3 = 2*(p+1)
            def Poussin(x):
                return np.sin(np.pi*(c1*x))*np.sin(np.pi*(c2*x))/(c3*np.sin(np.pi*(x/2))**2)
            return Poussin

        def PoussinAbsWrapper(n,p):
            c1 = (2*n+1-p)/2
            c2 = (p+1)/2
            c3 = 2*(p+1)
            def Poussin(x):
                return np.abs(np.sin(np.pi*(c1*x))*np.sin(np.pi*(c2*x))/(c3*np.sin(np.pi*(x/2))**2))
            return Poussin

        f = PoussinAbsWrapper(self.n_pous,self.n_pous//2)
        g = PoussinWrapper(self.n_pous,self.n_pous//2)


        self.noises = np.zeros(self.n_max)
        self.sig = np.zeros_like(self.noises)
        self.noises[0] = np.random.uniform(-0.1,0.1)

        for i in range(self.n_max-1):
            x1 = np.random.uniform(-1,1)
            q = f(x1)/f(self.noises[i])
            if np.random.binomial(1,min(q,1)) == 1:
                self.noises[i+1] = x1
            else:
                self.noises[i+1] = self.noises[i]

        for i in range(self.n_max):
            self.sig[i] = 2*(g(self.noises[i])>0)-1

    def adjust(self, x):
        if x > 1:
            return x-2
        if x < -1:
            return x+2
        return x