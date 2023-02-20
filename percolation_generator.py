# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 02:23:10 2023

@author: kevin
"""
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import scipy.optimize as op

params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [15,15],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

#%% Naming the file to save 
fname = 'percolation_data_prelim_03'

def file_id(name, pkl = True, directory = None):
    """
    Returns:
        Returns the file name with all the relevant directories
    """
    if directory == None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # dir_path = os.getcwd()
        directory = dir_path#os.path.dirname(dir_path)
    else:
        directory = directory
    if pkl == True:
        pkl = 'pkl'
    else:
        pkl = pkl
    __file_name = f'{name}'
    _file_name = str(__file_name).replace(' ', '-').replace(',', '').replace('[', '-').replace(']','-').replace('.','-')
    file_name = os.path.join(directory, 'DAG_data_files\percolation_data', f'{_file_name}.pkl')
    return file_name
print(file_id(fname))

#%% Set up parameters of the simulation
D = [1,2,3]
P = 2
V = 1
RHO = [2**9, 2**10, 2**11, 2**12]
M = 1000
# K = np.array([(2+i/50) for i in range(-10,10)])
K = np.array([0.1,1,2,3,4,5])
#%% Data generator; tries to retrieve existing data first 
try:
    dataframe = pickle.load(open(f'{file_id(fname)}', 'rb'))
except:
    dataframe = {d:{k:{rho: {'r':pa.convert_degree_to_radius(k, rho, d, P), 'p':[], 'sc':[], 'gwcc':[]} for rho in RHO} for k in K} for d in D}
    for i in range(M):
        print(f"""
              (⌐▨_▨)     BEGIN ITERATION: {i}""")
        for d in D:      
            for rho in RHO:
                percolating = False
                pos = rgg._poisson_cube_sprinkling(rho, V, d, fixed_N = True)
                for k in K:
                    if percolating != 2:
                        r = dataframe[d][k][rho]['r']
                        edge_list, graph_dict = rgg.lp_random_geometric_graph(pos, r, P, show_dist = False)
                        percolating = pa.DFS_percolating(graph_dict)
                    else:
                        None
                    dataframe[d][k][rho]['p'].append(percolating)
                    print(len(edge_list)/rho)
#%% Save file
f = open(f'{file_id(fname)}', 'wb')
pickle.dump(dataframe, f)
f.close()
#%%
y = {d: [np.sum(dataframe[d][k][RHO[0]]['p'])/M for k in K] for d in D}
x = K
for d in D:
    plt.plot(x,y[d], label = 'd = %s' % (d))

plt.show()

#%% Plotting the Bastas plot
colours = iter(['red', 'orange', 'gold', 'green', 'teal', 'dodgerblue', 'blue', 'purple', 'magenta'])
def power_law(x, a, b):
    return a * x ** b

def power_fit(x, y, **kwargs):
    u = np.linspace(x[0], x[-1], 1000)
    params, cov = op.curve_fit(power_law, x, y, **kwargs)
    
    v = power_law(u, *params)
    return u, v, params, cov
#%%
for d in D:
    for k in K[:]:
        colour = next(colours)
        y = [np.sum(dataframe[d][k][rho]['p'])/M for rho in RHO]
        yerr = [np.sqrt(M*Y*(1-Y))/M for Y in y]
        x = RHO
        u, v, params, cov = power_fit(x, y, sigma = yerr)
        
        plt.errorbar(x, y, yerr, capsize = 10, fmt = 'x', color = colour)
        plt.plot(u, v, color = colour, label = f'k = {k} $\pm$ {np.round(np.sqrt(cov[1][1]), 3)}')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$\Pi$')
        plt.legend()
        plt.title('D = %s' % (d)) #only needed for dimensional plots
        print(params)
    plt.show()

#%% Record of preliminary data
"""
percolation_data_prelim_01: 
    RHO = [250, 500, 1000, 2000]
    K = np.array([(2+i/50) for i in range(-1,8)])
    
percolation_data_prelim_02:
    RHO = [250, 500, 1000, 2000]
    K = np.array([(2+i/50) for i in range(2,10)])

percolation_data_prelim_03:
    RHO = [2**9, 2**10, 2**11, 2**12]
    K = np.array([0.1,1,2,3,4,5])
    M = 1000

For d=2:
    >>> Searching for the phase transition
    RHO = [2**9, 2**10, 2**11, 2**12]
    K = np.array([(2+i/50) for i in range(0,15)])
    
    >>> Wide spanning search to see the entire behaviour
    RHO = [2**9, 2**10, 2**11, 2**12]
    K = np.array([(1 + i/10) for i in range(0,30)])
"""