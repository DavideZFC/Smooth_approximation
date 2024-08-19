import numpy as np
from classes.environment.real_curve_fit import real_curve_fit
from classes.learner.my_linear_models.poussin import *
from classes.learner.baselines.KDE import *
from functions.misc.make_experiment import make_experiment

idx = 48
y = np.load('data\DuaLipaHoudini\{}.npy'.format(idx))
y = np.load('data\DuaLipaHoudini\my_houdini_intro.npy')[:500]
n = 100000
seeds = 5

env = real_curve_fit(y=y,sigma=0.1)

agent1 = Poussin('Fourier', 1000, d=100, optimal_deisgn=True, new_idea=True, n_max=n, n_pous=24)
agent2 =  MetaLearner('Poly', 1000, d=24, optimal_deisgn=True, new_idea=True, n_max=n)
agent3 = KDE(h=0.0001)

label1 = 'Pous200'
label2 = 'Poly24'
label3 = 'KDE0.05'

exp_name = 'Poisson_vs_polinomi_vs_KDE'
make_experiment([agent1,agent2, agent3], env, seeds=seeds, n=n, labels=[label1, label2, label3], exp_name=exp_name)

