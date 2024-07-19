import numpy as np
from copy import deepcopy


def feed_with_samples(env, learner, n):
    for _ in range(n):
        x = learner.query()
        y = env.get_sample(x)
        learner.update(y)

def MSE(y1,y2):
    y = y1-y2
    return np.mean(y**2)**0.5

def infty(y1,y2):
    y = y1-y2
    return np.max(np.abs(y))

def test_algorithm(agent0, env, n=1000, seeds=1, first_seed=1):
    '''
    Test a given policy on an environment and returns the regret estimated over some different random seeds

    Parameters:
        agent (specific class): policy to be tested
        env (class environment): environment over which to test the policy
        n (int): number of samples
        seeds (int): how many random seeds to use
        first seed (int): first seed to use

    Returns:
        curve_matrix (array): matrix having as rows the value of the curve estimated for each one random seed
    '''

    np.random.seed(first_seed)
    y_true = np.copy(env.y)
    N = len(y_true)
    x = np.linspace(-1,1,N)

    L2err = np.zeros(seeds)
    Linferr = np.zeros(seeds)
    prediction_matrix = np.zeros((seeds,N))

    for seed in range(seeds):
        agent = deepcopy(agent0)

        env.seed(seed)
        feed_with_samples(env, agent, n=n)

        agent.train_model()
        y_pred = agent.predict(x)

        L2err[seed] = MSE(y_true,y_pred)
        Linferr[seed] = infty(y_true,y_pred)
        prediction_matrix[seed,:] = y_pred

    return L2err, Linferr, prediction_matrix

    

        

