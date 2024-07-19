import numpy as np

class real_curve_fit:
    '''
    Class to make curve fit
    '''

    def __init__(self, y, sigma=0.5, seed = 257):
        '''
        Defines the variables in the environment

        Parameters:
        sigma (double): standard deviation of the noise
        curve (str): name of the reward curve to build. Possible values:
            'gaussian',
        seed (int): random seed to fix
        '''

        np.random.seed(seed)

        # standard deviation of the noise
        self.sigma = sigma

        # make curve
        self.y = y
        self.L = len(y)
        
    def seed(self, s):
        np.random.seed(seed=s)
    
    def get_sample(self, x):
        '''
        Get a sample from the curve
        '''
        x_resc = (x + 1)/2*(self.L-1)+np.random.uniform()
        
        noise = np.random.normal(0,self.sigma)
        return self.y[int(x_resc)] + noise