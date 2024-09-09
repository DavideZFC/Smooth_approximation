import os
import datetime
import time
import json
import numpy as np
from functions.misc.test_algorithm import test_algorithm
from functions.misc.plot_data import plot_data
from functions.misc.confidence_bounds import bootstrap_ci
import matplotlib.pyplot as plt
from tqdm import tqdm


def save_parameters(policies, env, n_vec, labels, dir):
    params = {}
    np.save(dir+'n_samples', n_vec)
    params['sig'] = env.sigma
    with open(dir+"env.json", "w") as f:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(params, f)

    for i in range(len(policies)):
        par = policies[i].get_data()
        with open(dir+"{}.json".format(labels[i]), "w") as f:
            # Convert the dictionary to a JSON string and write it to the file
            json.dump(par, f)



def make_experiment(policies, env, seeds, n_vec, nu, norm_est, sigma, labels, exp_name='', save=True):
    '''
    Performs a RL experiment, estimating the reward curve and saving the data in a given folder

    Parameters:
        policies (list): list of policies to be tested
        env (class environment): environment over which to test the policies
        n (int): number of samples
        seeds (int): how many random seed to use in the experiment
        labels (list): list with the same length of policies giving a name to each one
        exp_name (string): string to be added to the filder created to same the data    
    '''

    # create folder
    if save:
        tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
        dir = 'results/'+'_'+tail+exp_name

        os.mkdir(dir)
        dir = dir+'/'
        save_parameters(policies, env, n_vec, labels, dir)

    # in this dictionary, we store the running times of the algorithms
    running_times = {}
    errors = {}

    y_true = env.y
    N = len(y_true)
    x = np.linspace(-1,1,N)
    
    for i in tqdm(range(len(policies))):

        running_times[labels[i]] = np.zeros(len(n_vec))
        errors[labels[i]] = np.zeros((seeds, len(n_vec)))

        for j in range(len(n_vec)):

            n = n_vec[j]
            policies[i].adjust_params(nu, norm_est, n, sigma)

            ####################################
            # actual algorithm simulation

            # evaluate running time of the algorithm
            t0 = time.time()

            # test the algorithm

            L2err, Linferr, prediction_matrix = test_algorithm(policies[i], env, n=n, seeds=seeds)

            # store time
            t1 = time.time()
            running_times[labels[i]][j] = t1 - t0
            errors[labels[i]][:,j] = Linferr

    if save:
        for i in range(len(policies)):
            
            np.save(dir+labels[i], errors[labels[i]])

            # make nonparametric confidence intervals
            low, high = bootstrap_ci(errors[labels[i]])

            # make plot of the error
            plot_data(n_vec, low, high, col='C{}'.format(i+1), label=labels[i])
            
        plt.legend()
        plt.savefig(dir+'errors.pdf')
        plt.clf() 

        for i in range(len(policies)):
            plt.plot(n_vec, running_times[labels[i]], marker='H', color='C{}'.format(i+1), label=labels[i])
            np.save(dir+labels[i]+'running', running_times[labels[i]])

        plt.yscale('log')
        plt.legend()
        plt.savefig(dir+'times.pdf')

    return 0

    