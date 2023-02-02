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
RHO = 2000 #density
V = 1 #volume
D = 2 #dimensions
K = 3 #average expected degree
M = 20 #number of graphs to generate (i.e. number of iterations)
#%% Measurement variables
_var = ['d', 'j1', 'j2', 'l']
_path_label = ['_sp', '_lp', '_gp']
var = [v + p for v in _var for p in _path_label]
_col = ['green', 'blue', 'red']

#%%
a = np.sqrt(2)
P = list(np.round([1/a**4, 1/a**3, 1/a**2, 1/a, 1, a, a**2, a**3, a**4], decimals = 5))

path_measures = {p:dict.fromkeys(var, 0) for p in P for v in var}
path_measures = {v:{p:[] for p in P} for v in var}
#%%
for i in range(M):
    print(i)
    _P = {p:{} for p in P}
    while _P:
        print('perc')
        pos = rgg._poisson_cube_sprinkling(RHO, V, D, fixed_N = True)
        _P = {p:{} for p in P}
        for p in P:
            r = pa.convert_degree_to_radius(K, RHO, D, p)
            edge_list, graph_dict, pos = rgg.lp_random_geometric_graph(pos, r, p)
            percolating = pa.DFS_percolating(graph_dict)
            if percolating == True:
                _P.pop(p)
        
    for p in P:
        r = pa.convert_degree_to_radius(K, RHO, D, p)
        edge_list, graph_dict, pos = rgg.lp_random_geometric_graph(pos, r, p)
        
        sp, lp = pa.short_long_paths(graph_dict)
        gp = pa.greedy_path(graph_dict)
        paths = [sp, lp, gp]
        
        for i in range(len(paths)):
            _d, _l = pa.pathDist(graph_dict, paths[i], p)
            path_measures[_var[0]+_path_label[i]][p].append(_d)
            path_measures[_var[3]+_path_label[i]][p].append(_l)
            path_measures[_var[1]+_path_label[i]][p].append(pa.pathJaggy(graph_dict, pos, paths[i]))
            path_measures[_var[2]+_path_label[i]][p].append(pa.pathJaggy2(graph_dict, pos, paths[i]))
        
        
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
for v in _var:
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
    plt.show()

# plt.hist(jaggy_list[0.25][0])
        