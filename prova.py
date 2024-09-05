import numpy as np
from classes.environment.real_curve_fit import real_curve_fit
from classes.learner.my_linear_models.poussin import *
from classes.learner.baselines.KDE import *
from classes.learner.baselines.NW_estimator import *
from functions.misc.make_experiment import make_experiment

save = False
idx = 48
sigma=0.1
y = np.load('data\DuaLipaHoudini\{}.npy'.format(idx))
# y = np.load('data\DuaLipaHoudini\my_houdini_intro.npy')[:500]
n = 1000
seeds = 5

env = real_curve_fit(y=y,sigma=sigma)

##### compute order-optimal parameters
nu = 5
norm_est = 100000
h = norm_est**(-2/(2*nu+1))*n**(-1/(2*nu+1))
d = int(n**(1/(2*nu+1))*norm_est**(2/(2*nu+1))*sigma**(-2/(2*nu+1)))

print('h = {}, d = {}'.format(h,d))


agent1 = NW(h=h)# Poussin('Fourier', 1000, d=100, optimal_deisgn=True, new_idea=True, n_max=n, n_pous=24)
agent2 =  Poussin('Fourier', 1000, d=d, optimal_deisgn=True, new_idea=True, n_max=n, n_pous=24)# MetaLearner('Poly', 1000, d=24, optimal_deisgn=True, new_idea=True, n_max=n)
agent3 = KDE(h=h)

label1 = 'NW_{}_{}'.format(nu,round(h,3))
label2 = 'Poussin'+str(d)
label3 = 'KDE_'+str(round(h,3))

exp_name = 'NW_vs_KDE'
results_pol = make_experiment([agent1,agent2, agent3], env, seeds=seeds, n=n, labels=[label1, label2, label3], exp_name=exp_name, save=save)

print(results_pol)