from kan import KAN
import torch
import numpy as np

class KANregressor:
    def __init__(self, lay=4, a=1, b=1, n_max=10000):
        self.lay = lay
        self.a = a
        self.b = b

        self.queries = np.zeros(n_max)
        self.evals = np.zeros(n_max)
        self.idx = 0
        
    def query(self):
        self.queries[self.idx] = np.random.uniform(low=self.a, high=self.b)
        return self.queries[self.idx]
    
    def update(self, y):
        self.evals[self.idx] = y
        self.idx += 1

    def train_model(self):
        X = self.queries[:self.idx].reshape(-1,1)
        y = self.evals[:self.idx].reshape(-1,1)

        dataset = {}
        dataset['train_input'] = torch.from_numpy(X)
        dataset['train_label'] = torch.from_numpy(y)
        dataset['test_input'] = torch.from_numpy(X)
        dataset['test_label'] = torch.from_numpy(y)

        def train_err():
            return torch.mean((self.model(dataset['train_input']) - dataset['train_label']).float()**2)
        def test_err():
            return torch.mean((self.model(dataset['test_input']) - dataset['test_label']).float()**2)

        self.dataset = dataset

        self.model = KAN(width=[1,self.lay,1], grid=3, k=3)
        self.model.train(dataset, opt="LBFGS", steps=20, metrics=(train_err,test_err))

    def predict(self, x):
        self.train_model()

        Xtesttorch = torch.from_numpy(x.reshape(-1,1))
        ytorch = self.model(Xtesttorch)

        return ytorch.detach().numpy()