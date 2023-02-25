# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:01:46 2023

@author: ariel
"""

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from tqdm import tqdm
import copy 
import multiprocessing

fname = 'para_geo_500_10' #HPC_opt_data_rho_M

#%%
RHO = 500
V = 1
D = 2
K = 3
M = 5
#%%
dep_var = ['d', 'l','j3']
path_type = ['sp', 'lp', 'gp'] #['spg', 'lpg', 'gp'] or #['spn', 'lpn', 'gp']  #use __n for network optimization, __g for geometric optimization
optimizer = 'G'
a = np.sqrt(2)
b = 1.025
P = list(np.round([a**n for n in range(-4,5)], decimals = 5)) + list(np.round([b**n for n in range(-4,5)], decimals = 5))
P = list(set(P))
P.sort()
#%%
def generateDataframe(M = None):
    dataframe = {dv:{pt:{p:{'raw':[]} for p in P} for pt in path_type} for dv in dep_var}
    for p in P:
        for path in path_type:
            dataframe['d'][path][p]['raw'] = []
            dataframe['l'][path][p]['raw'] = []
            dataframe['j3'][path][p]['raw'] = []
            dataframe['j3'][path][p]['sum'] = []
            dataframe['j3'][path][p]['mean'] = []
            dataframe['j3'][path][p]['err'] = []

    if M != None:
        dataframe['config'] = {'constants': [RHO, V, D, K, M], 'dep_var': dep_var, 'path_types': path_type, 'optimizer': optimizer}
    
    return dataframe
#%%
df = generateDataframe()
#%%
def geo_generator():
    dataframe = generateDataframe()

    _P = {p:{} for p in P}
    G = {p:{'graph_dict':{}, 'edge_list':{}} for p in P}
    while _P:
        pos = rgg._poisson_cube_sprinkling(RHO, V, D, fixed_N = True)
        #_P = {p:{} for p in P}
        #G = {p:{'graph_dict':{}, 'edge_list':{}} for p in P}
        _PK = copy.deepcopy(list(_P.keys()))
        for p in _PK:
            r = pa.convert_degree_to_radius(K, RHO, D, p)
            edge_list, graph_dict = rgg.lp_random_geometric_graph(pos, r, p, show_dist = True)
            percolating = pa.DFS_percolating(graph_dict)
            if percolating == True:
                G[p]['graph_dict'] = graph_dict
                G[p]['edge_list'] = edge_list
                _P.pop(p)

    for p in P:
        edge_list = G[p]['edge_list']
        graph_dict = G[p]['graph_dict']

        sp, lp = pa.getPaths(graph_dict, 'geo')
        gp = pa.greedy_path_geo(graph_dict)
        paths = [sp, lp, gp] 
        paths = {path_type[i]: paths[i] for i in range(len(paths))}

        for path in path_type:
            _d, _l = pa.pathDist(graph_dict, paths[path], p)
            _J3 = pa.pathJaggy3(pos, paths[path])
            
            dataframe['d'][path][p]['raw'] = _d
            dataframe['l'][path][p]['raw'] = _l

            #dataframes take angular all angular values in the form (angle list, sum, mean, std)
            dataframe['j3'][path][p]['raw'] = _J3[0]
            dataframe['j3'][path][p]['sum'] = _J3[1]
            dataframe['j3'][path][p]['mean'] = _J3[2]
            dataframe['j3'][path][p]['err'] = _J3[3]

    return dataframe
#%%
df = geo_generator()
#%%
x = np.linspace(0,1, 100)
def f():  # no argument
    return x*x
#%%
start = time.perf_counter()
if __name__ == "__main__":
    print("""
          -----------------------------
          
              STARTING MULTIPROCESS
          
          -----------------------------
          """)
    pool = multiprocessing.Pool(2)
    result2 = pool.starmap(geo_generator, [() for _ in range(M)]) #uses all available processors
    pool.close()
    pool.join()
    
#%%
with open('DAG_data_files\percolation_data\percolation_data_HPC_TEST.pkl', 'rb') as handle:
    b = pickle.load(handle)
