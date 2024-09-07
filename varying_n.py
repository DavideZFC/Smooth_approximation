import numpy as np
from classes.environment.real_curve_fit import real_curve_fit
from classes.learner.my_linear_models.poussin import *
from classes.learner.baselines.KDE import *
from classes.learner.baselines.NW_estimator import *
from functions.misc.change_n_experiment import make_experiment

save = True
idx = 50
sigma=0.1
y = np.load('data\DuaLipaHoudini\{}.npy'.format(idx))
n_vec = np.array([100, 200, 500, 1000],dtype=np.int16)
seeds = 5

env = real_curve_fit(y=y,sigma=sigma)

##### compute order-optimal parameters
nu = 4
norm_est = 8000
h = norm_est**(-2/(2*nu+1))*n_vec[0]**(-1/(2*nu+1))
d = int(n_vec[0]**(1/(2*nu+1))*norm_est**(2/(2*nu+1))*sigma**(-2/(2*nu+1)))

print('h = {}, d = {}'.format(h,d))


agent1 = NW(h=h)# Poussin('Fourier', 1000, d=100, optimal_deisgn=True, new_idea=True, n_max=n, n_pous=24)
print(agent1.nu, agent1.h)

agent2 =  Poussin('Fourier', 1000, d=d, optimal_deisgn=True, new_idea=True, n_max=n_vec[-1], n_pous=d)# MetaLearner('Poly', 1000, d=24, optimal_deisgn=True, new_idea=True, n_max=n)
agent3 = KDE(h=h)

label1 = 'NW_{}_{}'.format(nu,round(h,3))
label2 = 'Poussin'+str(d)
label3 = 'KDE_'+str(round(h,3))

exp_name = 'Tutti contro tutti'
make_experiment([agent1,agent2, agent3], env, seeds=seeds, n_vec=n_vec, nu = nu,
norm_est=norm_est, sigma=sigma, labels=[label1, label2, label3], exp_name=exp_name, save=save)
