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

params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [15,15],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

#%%
fname = 'percolation_data_prelim_01'
#%%
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

#%%
D = 2
P = 2
V = 1
RHO = [250, 500, 1000, 2000]
M = 1000
K = np.array([(2+i/50) for i in range(-1,8)])
#%%
dataframe = {k:{rho: {'r':pa.convert_degree_to_radius(k, rho, D, P), 'p':[]} for rho in RHO} for k in K}
#%%
try:
    dataframe = pickle.load(open(f'{file_id(fname)}', 'rb'))
except:
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
#%%
f = open(f'{file_id(fname)}', 'wb')
pickle.dump(dataframe, f)
f.close()
#%%
for k in K:
    y = [np.sum(dataframe[k][rho]['p'])/M for rho in RHO]
    yerr = []
    x = RHO
    
    # y = [np.sum(dataframe[r]['p'])/M for r in R]
    # y = [np.average(dataframe[r]['k']) for r in R]
    plt.plot(x, y)
    plt.xscale('log')
    plt.yscale('log')