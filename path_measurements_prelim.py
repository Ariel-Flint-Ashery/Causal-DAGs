# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:15:33 2023

@author: kevin
"""

import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import matplotlib.pyplot as plt
import numpy as np

#%% Independent Variable 
RHO = 2000
V = 1
D = 2
K = 3
M = 20
#%% Measurement variables
dep_var = ['d', 'j1', 'j2', 'l']
path_type = ['_sp', '_lp', '_gp']
optimizer = 'net' #or 'geo'
a = np.sqrt(2)
P = list(np.round([1/a**4, 1/a**3, 1/a**2, 1/a, 1, a, a**2, a**3, a**4], decimals = 5))

dataframe = {dv:{pt:{p:[] for p in P} for pt in path_type} for dv in dep_var}
#%% plotting params
params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [11,11],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

_col = ['green', 'blue', 'red']
#%%
for i in range(M):
    print(i)
    _P = {p:{} for p in P}
    POS = {p:None for p in P}
    print('Percolating...')
    while _P:
        pos = rgg._poisson_cube_sprinkling(RHO, V, D, fixed_N = True)
        _P = {p:{} for p in P}
        for p in P:
            r = pa.convert_degree_to_radius(K, RHO, D, p)
            edge_list, graph_dict = rgg.lp_random_geometric_graph(pos, r, p)
            percolating = pa.DFS_percolating(graph_dict)
            if percolating == True:
                POS[p] = pos
                _P.pop(p)
    print("""
    -----------------------------
        STARTING MEASUREMENTS
    -----------------------------
    """)    
    for p in P:
        r = pa.convert_degree_to_radius(K, RHO, D, p)
        edge_list, graph_dict = rgg.lp_random_geometric_graph(POS[p], r, p)
        
        if optimizer = 'net':
            sp, lp = pa.short_long_paths(graph_dict) #I think Kevin's algorith is faster for network paths.
        if optimizer = 'geo':
            sp, lp = pa.getPaths(graph_dict, 'geo')
        #greey path works for network optimization only!
        gp = pa.greedy_path(graph_dict)
        paths = [sp, lp, gp]
        paths = {path_type[i]: paths[i] for i in range(len(paths))}
        
        for path in path_type:
            _d, _l = pa.pathDist(graph_dict, paths[path], p)
            dataframe['d'][path][p].append(_d)
            dataframe['l'][path][p].append(_l)
            dataframe['j1'][path][p].append(pa.pathJaggy(graph_dict, pos, paths[path]))
            dataframe['j2'][path][p].append(pa.pathJaggy2(graph_dict, pos, paths[path]))

        #calculate errors
        for path in path_type:
            dataframe['d_err'][path][p] = np.std(dataframe['d'][path][p], ddof = 1) #ddof = 1 since we are sampling from the inifinite graph ensemble
            dataframe['l_err'][path][p] = np.std(dataframe['l'][path][p], ddof = 1)
            dataframe['j1_err'][path][p] = np.std(dataframe['j1'][path][p], ddof = 1)/np.sqrt(M) #not correct! need to use std of each angle average
            dataframe['j2_err'][path][p] = np.std(dataframe['j2'][path][p], ddof = 1)/np.sqrt(M)
        
        # spd = pa.pathDist(graph_dict, sp, p)
        # lpd = pa.pathDist(graph_dict, lp, p)
        # gpd = pa.pathDist(graph_dict, gp, p)
        
        # spj1 = pa.pathJaggy(graph_dict, pos, sp)
        # lpj1 = pa.pathJaggy(graph_dict, pos, lp)
        # gpj1 = pa.pathJaggy(graph_dict, pos, gp)
        
        # spj2 = pa.pathJaggy2(graph_dict, pos, sp)
        # lpj2 = pa.pathJaggy2(graph_dict, pos, lp)
        # gpj2 = pa.pathJaggy2(graph_dict, pos, gp)
#%%
for v in dep_var[0]:
    col = iter(_col)
    for l in _path_label[:-1]:
        colour = next(col)
        x = P
        _y = list(path_measures[v+l].values())
        y = [np.average(j) for j in _y]
        yerr = [np.std(j) for j in _y]
        plt.plot(x, y, color = colour)
        plt.errorbar(x, y, yerr = yerr, label = v+l, fmt = '.', ms = 20, capsize = 10, color = colour)
    plt.xlabel('p')
    plt.ylabel(v)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
#%%
jaggy_dict = path_measures['j1_sp']
jaggy_list = {p:{i:jaggy_dict[p][i][0] for i in range(M)} for p in P}

for p in P:
    J = []
    for l in jaggy_list[p].values():
        J = J + l
    plt.hist(J, bins = 5)
    plt.title('j1_sp ' + 'p=' + str(p))
    plt.show()
#%%
jaggy_dict = path_measures['j1_lp']
jaggy_list = {p:{i:jaggy_dict[p][i][0] for i in range(M)} for p in P}

for p in P:
    J = []
    for l in jaggy_list[p].values():
        J = J + l
    plt.hist(J, bins = 5)
    plt.title('j1_lp  '+'p='+str(p))
    plt.show()
        