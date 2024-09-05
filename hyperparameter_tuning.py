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
nu = 4+np.arange(3,dtype=np.int64)
print(nu)
norm_est = 2**np.linspace(5,10,5)


policies = []
labels = []

### create some NW policies
for i in range(len(nu)):
    h = norm_est**(-2/(2*nu[i]+1))*n**(-1/(2*nu[i]+1))
    policies.append(NW(h=h,nu=nu[i]))
    labels.append('NW_nu={}'.format(nu[i]))

print('NW policies ready')

### create some KDE policies
for i in range(len(nu)):
    h = norm_est**(-2/(2*nu[i]+1))*n**(-1/(2*nu[i]+1))
    policies.append(KDE(h=h))
    labels.append('KDE_nu={}'.format(nu[i]))

print('KDE policies ready')

### create some Poussin policies
for i in range(len(nu)):
    for j in range(len(norm_est)):
        h = norm_est**(-2/(2*nu[i]+1))*n**(-1/(2*nu[i]+1))
        d = int(n**(1/(2*nu[i]+1))*norm_est[j]**(2/(2*nu[i]+1))*sigma**(-2/(2*nu[i]+1)))
        policies.append(Poussin('Fourier', 1000, d=d, optimal_deisgn=True, new_idea=True, n_max=n, n_pous=d))
        labels.append('Poussin_nu_{}_norm_{}'.format(nu[i],int(norm_est[j])))

print('Poussin policies ready')     

print(labels)
exp_name = 'KKK'
results_pol = make_experiment(policies, env, seeds=seeds, n=n, labels=labels, exp_name=exp_name, save=save)
