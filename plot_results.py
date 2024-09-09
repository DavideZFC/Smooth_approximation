import matplotlib.pyplot as plt
import json
import numpy as np
from functions.misc.confidence_bounds import bootstrap_ci
from functions.misc.plot_data import plot_data
import os

def filter_and_save(x, v1, v2, dir, filter=1):
    v1 = v1[::filter]
    v2 = v2[::filter]
    x = x[::filter]
    names = ['mean', 'low', 'up']
    for name in names:
        if name == 'mean':
            mat = np.column_stack((x,(v1+v2)/2))
        elif name == 'low':
            mat = np.column_stack((x,v1))
        elif name == 'up':
            mat = np.column_stack((x,v2))
        name = dir+'/'+name+'.txt'
        np.savetxt(name, mat)

def plot_label(label, color, filename = 'TeX/template_plot.txt'):
    with open(filename, 'r') as file:
        # read in the contents of the file
        contents = file.read()
    file.close()

    # replace all occurrences of 'H' with 'my_word'
    contents = contents.replace('H', label)

    # replace all occurrences of 'K' with 'my_other_word'
    contents = contents.replace('K', color)

    return contents

def add_file(filename='TeX/reference_tex.txt'):    
    with open(filename, 'r') as file:
        content = file.read()
    file.close()
    return content


dir = 'results\_24_09_09-14_58_Tutti contro tutti'
labels = ['LPE', 'NW', 'Poussin']# lista nomi CHE NON POSSONO AVERE H o K
true_lab = ['LPE', 'NW', 'Poussin']
print(labels)


new_dir = dir+'/TeX'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

mid_dir = new_dir + '/data'
if not os.path.exists(mid_dir):
    os.mkdir(mid_dir)

c = 0

with open(new_dir+'/main.txt', 'w') as new_file:
    new_file.write(add_file())


n_samples = np.load(dir+'/n_samples.npy')
for l in labels:
    results = np.load(dir+'/'+l+'.npy')
    
    # make nonparametric confidence intervals
    low, high = bootstrap_ci(results, resamples=1000)
    T = len(low)
    color = 'C{}'.format(c)
    true_lab = l.replace('_','')

    # make plot
    plot_data(n_samples, low, high, col=color, label=true_lab)

    # update color
    c += 1

    # in this part, we crea new folders to save all the necessary info
    this_dir = mid_dir + '/' + true_lab
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)
    filter_and_save(n_samples, low, high, this_dir, filter=1)

    # second part: the running times
    running = np.load(dir+'/'+l+'running'+'.npy')
    mat = np.column_stack((n_samples,running))
    name = this_dir+'/'+l+'running'+'.txt'
    np.savetxt(name, mat)


    print(l+' done')

c = 0
for l in labels:
    color = 'C{}'.format(c)
    c += 1
    true_lab = l.replace('_','')
    with open(new_dir+'/main.txt', 'a') as new_file:
        new_file.write(plot_label(true_lab, color))

c = 0
for l in labels:
    color = 'C{}'.format(c)
    c += 1
    true_lab = l.replace('_','')
    with open(new_dir+'/main.txt', 'a') as new_file:
        new_file.write(plot_label(true_lab, color, filename = 'TeX/template_fill.txt'))

with open(new_dir+'/main.txt', 'a') as new_file:
    new_file.write(add_file('TeX/refrence_end.txt'))

### passiamo a running times
with open(new_dir+'/running_times.txt', 'w') as filex:
    filex.write(add_file())

c = 0
for l in labels:
    color = 'C{}'.format(c)
    c += 1
    true_lab = l.replace('_','')
    with open(new_dir+'/running_times.txt', 'a') as filex:
        filex.write(plot_label(true_lab+'running', color))

with open(new_dir+'/running_times.txt', 'a') as filex:
    filex.write(add_file('TeX/refrence_end.txt'))

plt.legend()
plt.title('learning_curves')
plt.savefig(dir+'/learning_curves.pdf')



