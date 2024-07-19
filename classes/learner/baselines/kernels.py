import numpy as np

def gaussian_kernel(x,h):
    return np.exp(-(x/h)**2)