import numpy as np
import math

def comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def make_cosin_arms(n_arms, d, arms):
    linUCBarms = np.zeros((n_arms, d))

    # build linear features from the arms
    for j in range(d):
        linUCBarms[:,j] = np.cos(np.pi*j*arms)

    return linUCBarms

def make_sincos_arms(n_arms, d, arms):
    linUCBarms = np.zeros((n_arms, d))

    # build linear features from the arms
    for j in range(d):
        if j == 0:
            # the constant tem is normalized to have L2 norm equal to one on [-1, 1]
            linUCBarms[:,j] = (1/2)**0.5
        elif j%2 == 1:
            eta = j//2+1
            linUCBarms[:,j] = np.sin(np.pi*eta*arms)
        else:
            eta = j//2
            linUCBarms[:,j] = np.cos(np.pi*eta*arms)

    return linUCBarms


def apply_poly(coef, x):
    ''' Returns the result gotten by applying a poplynomial with given coefficients to
    a vector of pints x'''

    # polynomial degree (+1)
    degree = len(coef)

    # number of x
    N = len(x)

    # store vector of powers of x
    x_pow = np.zeros((N, degree))

    for i in range(degree):
        if i>0:
            x_pow[:,i] = x**(i)
        else:
            x_pow[:,i] = 1
    
    return np.matmul(x_pow, coef.reshape(-1,1)).reshape(-1)


def computeL2(coef):
    N = 1000
    x = np.linspace(-1,1,N)
    y = apply_poly(coef, x)

    return (2*np.mean(y**2))**0.5

def get_legendre_poly(n):
    ''' get the coefficients of the n-th legendre polynomial'''

    coef = np.zeros(n+1)
    for k in range(int(n/2)+1):
        coef[n-2*k] = (-1)**k * 2**(-n) * comb(n, k) * comb (2*n-2*k, n)
    return coef


def get_legendre_norm_poly(n):
    ''' get the coefficients of the n-th legendre polynomial'''

    coef = np.zeros(n+1)
    for k in range(int(n/2)+1):
        coef[n-2*k] = (-1)**k * 2**(-n) * comb(n, k) * comb (2*n-2*k, n)
    
    return coef/computeL2(coef)



def make_legendre_arms(n_arms, d, arms):
    linUCBarms = np.zeros((n_arms, d))

    # build linear features from the arms
    for j in range(d):
        
        # compute degree j legendre polynomial
        coef = get_legendre_poly(j)
        
        # apply polynomial to the arms
        linUCBarms[:,j] = apply_poly(coef, arms)
    
    return linUCBarms


def make_legendre_norm_arms(n_arms, d, arms):
    linUCBarms = np.zeros((n_arms, d))

    # build linear features from the arms
    for j in range(d):
        
        # compute degree j legendre polynomial
        coef = get_legendre_norm_poly(j)
        
        # apply polynomial to the arms
        linUCBarms[:,j] = apply_poly(coef, arms)
    
    return linUCBarms


def make_legendre_even_arms(n_arms, d, arms):
    linUCBarms = np.zeros((n_arms, d))

    # build linear features from the arms
    for j in range(d):
        
        # compute degree 2j legendre polynomial (not j, here we want only even polynomials)
        coef = get_legendre_poly(2*j)
        
        # apply polynomial to the arms
        linUCBarms[:,j] = apply_poly(coef, arms)

    return linUCBarms


def make_chebishev_arms(n_arms, d, arms):
    linUCBarms = np.zeros((n_arms, d))

    # build linear features from the arms
    for j in range(d):
        linUCBarms[:,j] = np.cos(j*np.arccos(arms))
    
    return linUCBarms


def make_chebishev_even_arms(n_arms, d, arms):
    linUCBarms = np.zeros((n_arms, d))

    # build linear features from the arms
    for j in range(d):
        linUCBarms[:,j] = np.cos((2*j)*np.arccos(arms))
    
    return linUCBarms


def make_poly_arms(n_arms, d, arms):
    linUCBarms = np.zeros((n_arms, d))

    # build linear features from the arms
    for j in range(d):
        
        # compute degree j standard polynomial
        coef = np.zeros(d+1)
        coef[j] = 1.
        
        # apply polynomial to the arms
        linUCBarms[:,j] = apply_poly(coef, arms)
    
    return linUCBarms
