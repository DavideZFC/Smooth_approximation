import os
import datetime
import time
import json
import numpy as np
from functions.misc.test_algorithm import test_algorithm
from functions.misc.plot_data import plot_data
from functions.misc.confidence_bounds import bootstrap_ci
import matplotlib.pyplot as plt


def save_parameters(policies, env, n, labels, dir):
    params = {}
    params['n'] = n
    params['sig'] = env.sigma
    with open(dir+"env.json", "w") as f:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(params, f)

    for i in range(len(policies)):
        par = policies[i].get_data()
        with open(dir+"{}.json".format(labels[i]), "w") as f:
            # Convert the dictionary to a JSON string and write it to the file
            json.dump(par, f)



def make_experiment(policies, env, seeds, n, labels, exp_name='', save=True):
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
        save_parameters(policies, env, n, labels, dir)

    # in this dictionary, we store the running times of the algorithms
    running_times = {}

    y_true = env.y
    N = len(y_true)
    x = np.linspace(-1,1,N)

    
    if save:
        plt.plot(x,y_true,label='True curve')
        np.save(dir+'true_curve',y_true)
    
    for i in range(len(policies)):

        ####################################
        # actual algorithm simulation

        # evaluate running time of the algorithm
        t0 = time.time()

        # test the algorithm

        L2err, Linferr, prediction_matrix = test_algorithm(policies[i], env, n=n, seeds=2)

        # store time
        t1 = time.time()
        running_times[labels[i]] = t1 - t0
        
        print(labels[i] + ' finished')

        if save:
            np.save(dir+labels[i], prediction_matrix)

            # make nonparametric confidence intervals
            low, high = bootstrap_ci(prediction_matrix)

            # make plot
            plot_data(x, low, high, col='C{}'.format(i+1), label=labels[i])

    if save:
        with open(dir+"running_times.json", "w") as f:
            # Convert the dictionary to a JSON string and write it to the file
            json.dump(running_times, f)
        
        plt.legend()
        plt.savefig(dir+'predictions.pdf')