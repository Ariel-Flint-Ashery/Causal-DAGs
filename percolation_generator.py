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
D = 2
P = 2
V = 1
RHO = [250, 500, 1000, 2000]
M = 1000
K = np.array([(2+i/50) for i in range(-10,10)])
#%% Data generator; tries to retrieve existing data first 
try:
    dataframe = pickle.load(open(f'{file_id(fname)}', 'rb'))
except:
    dataframe = {k:{rho: {'r':pa.convert_degree_to_radius(k, rho, D, P), 'p':[]} for rho in RHO} for k in K}
    for i in range(M):
        print(f"""
              (⌐▨_▨)     BEGIN ITERATION: {i}""")
        for rho in RHO:
            percolating = False
            pos = rgg._poisson_cube_sprinkling(rho, V, D, fixed_N = True)
            for k in K:
                if percolating == False:
                    r = dataframe[k][rho]['r']
                    edge_list, graph_dict = rgg.lp_random_geometric_graph(pos, r, P, show_dist = False)
                    percolating = pa.DFS_percolating(graph_dict)
                else:
                    None
                dataframe[k][rho]['p'].append(percolating)
#%% Save file
f = open(f'{file_id(fname)}', 'wb')
pickle.dump(dataframe, f)
f.close()
#%% Plotting
colours = iter(['red', 'orange', 'gold', 'green', 'teal', 'dodgerblue', 'blue', 'purple', 'magenta'])
def power_law(x, a, b):
    return a * x ** b

def power_fit(x, y, **kwargs):
    u = np.linspace(x[0], x[-1], 1000)
    params, cov = op.curve_fit(power_law, x, y, **kwargs)
    
    v = power_law(u, *params)
    return u, v, params, cov

for k in K[:]:
    colour = next(colours)
    y = [np.sum(dataframe[k][rho]['p'])/M for rho in RHO]
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
    print(params)

#%% Record of preliminary data
"""
percolation_data_prelim_01: 
    RHO = [250, 500, 1000, 2000]
    K = np.array([(2+i/50) for i in range(-1,8)])
    
percolation_data_prelim_02:
    RHO = [250, 500, 1000, 2000]
    K = np.array([(2+i/50) for i in range(2,10)])
"""