import numpy as np
from classes.environment.real_curve_fit import real_curve_fit
from classes.learner.my_linear_models.poussin import *
from classes.learner.baselines.KDE import *
from classes.learner.baselines.NW_estimator import *
from functions.misc.make_experiment import make_experiment
import pandas as pd

save = False
idx = 48
sigma=0.1
y = np.load('data\DuaLipaHoudini\{}.npy'.format(idx))
# y = np.load('data\DuaLipaHoudini\my_houdini_intro.npy')[:500]
n = 500
seeds = 2

env = real_curve_fit(y=y,sigma=sigma)

policies = []
labels = []

iter = 50
nu_list = []
norm_list = []

### create some Poussin policies
for i in range(iter):
    nu = 4+np.random.randint(8)
    norm_est = 2**np.random.uniform(5,25)
    nu_list.append(nu)
    norm_list.append(norm_est)

    h = norm_est**(-2/(2*nu+1))*n**(-1/(2*nu+1))
    policy = NW(h=h,nu=nu)
    policies.append(policy)
    labels.append('NW_nu={}'.format(nu))

    h = norm_est**(-2/(2*nu+1))*n**(-1/(2*nu+1))
    policies.append(KDE(h=h))
    labels.append('KDE_nu={}'.format(nu))


    h = norm_est**(-2/(2*nu+1))*n**(-1/(2*nu+1))
    d = int(n**(1/(2*nu+1))*norm_est**(2/(2*nu+1))*sigma**(-2/(2*nu+1)))
    policies.append(Poussin('Fourier', 1000, d=d, optimal_deisgn=True, new_idea=True, n_max=n, n_pous=d))
    labels.append('Poussin_nu_{}_norm_{}'.format(nu,int(norm_est)))


print(labels)
exp_name = 'KKK'
results_pol = make_experiment(policies, env, seeds=seeds, n=n, labels=labels, exp_name=exp_name, save=save)


NW_list = [results_pol[i] for i in range(len(results_pol)) if i % 3 == 0]
KDE_list = [results_pol[i] for i in range(len(results_pol)) if i % 3 == 1]
Poussin_list = [results_pol[i] for i in range(len(results_pol)) if i % 3 == 2]

dic = {}
dic['nu'] = nu_list
dic['norms'] = norm_list
dic['NW'] = NW_list
dic['KDE'] = KDE_list
dic['Poussin'] = Poussin_list

# Converti il dizionario in un DataFrame
df = pd.DataFrame(dic)

# Salva il DataFrame in un file Excel
df.to_excel('results/HP_tuning'+exp_name+'.xlsx', index=False)


