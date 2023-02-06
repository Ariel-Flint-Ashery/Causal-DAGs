# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:15:33 2023

@author: kevin
"""
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

fname = 'path_data_prelim_03'
#%%
def file_id(name, pkl = True, directory = None):
    """
    Returns:
        the file name for a given data set with parameters rho, v, d, p, iterations.
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
    file_name = os.path.join(directory, 'DAG_data_files\path_data', f'{_file_name}.pkl')
    return file_name
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
#%% Independent Variable
RHO = 1000
V = 1
D = 2
K = 3
M = 20 #500
#%% Measurement variables
dep_var = ['d', 'j1', 'j2', 'j3', 's1', 's2', 'l']
path_type = ['spg', 'lpg', 'gp'] #['spn', 'lpn', 'gp']  #use __n for network optimization, __g for geometric optimization
optimizer = 'geo' #'net' or 'geo'
a = np.sqrt(2)
P = list(np.round([1/a**4, 1/a**3, 1/a**2, 1/a, 1, a, a**2, a**3, a**4], decimals = 5))

dataframe = {dv:{pt:{p:{'raw':[]} for p in P} for pt in path_type} for dv in dep_var}
for v in dep_var[1:6]:
    for path in path_type:
        for p in P:
            dataframe[v][path][p]['mean'] = []
            dataframe[v][path][p]['err'] = []

for v in dep_var[1:4]:
    for path in path_type:
        for p in P:
            dataframe[v][path][p]['sum'] = []

dataframe['config'] = {'constants': [RHO, V, D, K, M], 'dep_var': dep_var, 'path_types': path_type, 'optimizer': optimizer}
#%%
try:
    dataframe = pickle.load(open(f'{file_id(fname)}', 'rb'))
except:
    for i in range(M):
        _P = {p:{} for p in P}
        POS = {p:None for p in P}
        print(f'Iteration {i}: Percolating...')
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

            if optimizer == 'net':
                sp, lp = pa.short_long_paths(graph_dict) #I think Kevin's algorith is faster for network paths.
            if optimizer == 'geo':
                sp, lp = pa.getPaths(graph_dict, 'geo')
            #greey path works for network optimization only!
            gp = pa.greedy_path(graph_dict)
            paths = [sp, lp, gp] 
            paths = {path_type[i]: paths[i] for i in range(len(paths))}

            for path in path_type:
                _d, _l = pa.pathDist(graph_dict, paths[path], p)
                _J1 = pa.pathJaggy(pos, paths[path])
                _J2 = pa.pathJaggy2(pos, paths[path])
                _J3 = pa.pathJaggy3(pos, paths[path])
                _S1 = pa.pathSeparation1(pos, paths[path])
                _S2 = pa.pathSeparation2(pos, paths[path], p)

                #dataframes take angular all angular values in the form (angle list, sum, mean, std)
                #theta, theta_m, theta_sum, theta_std = pa.pathJaggy(pos, paths[path])
                #theta, theta_m, theta_sum, theta_std = pa.pathJaggy2(pos, paths[path])
                #theta, theta_m, theta_sum, theta_std = pa.pathJaggy3(pos, paths[path])
                dataframe['d'][path][p]['raw'].append(_d)
                dataframe['l'][path][p]['raw'].append(_l)

                dataframe['j1'][path][p]['raw'].append(_J1[0])
                dataframe['j1'][path][p]['sum'].append(_J1[1])
                dataframe['j1'][path][p]['mean'].append(_J1[2])
                dataframe['j1'][path][p]['err'].append(_J1[3])

                dataframe['j2'][path][p]['raw'].append(_J2[0])
                dataframe['j2'][path][p]['sum'].append(_J2[1])
                dataframe['j2'][path][p]['mean'].append(_J2[2])
                dataframe['j2'][path][p]['err'].append(_J2[3])

                dataframe['j3'][path][p]['raw'].append(_J3[0])
                dataframe['j3'][path][p]['sum'].append(_J3[1])
                dataframe['j3'][path][p]['mean'].append(_J3[2])
                dataframe['j3'][path][p]['err'].append(_J3[3])

                dataframe['s1'][path][p]['raw'].append(_S1[0])
                dataframe['s1'][path][p]['mean'].append(_S1[1])
                dataframe['s1'][path][p]['err'].append(_S1[2])

                dataframe['s2'][path][p]['raw'].append(_S2[0])
                dataframe['s2'][path][p]['mean'].append(_S2[1])
                dataframe['s2'][path][p]['err'].append(_S2[2])
# %% Save file
f = open(f'{file_id(fname)}', 'wb')
pickle.dump(dataframe, f)
f.close()


#%%
#calculate errors and new measures
# for p in P:
#     for path in path_type:
#         dataframe['d_err'][path][p] = np.std(dataframe['d'][path][p], ddof = 1) #ddof = 1 since we are sampling from the inifinite graph ensemble
#         dataframe['l_err'][path][p] = np.std(dataframe['l'][path][p], ddof = 1)
#         std = np.average([np.std(angles, ddof = 1) for angles in dataframe['j1'][path][p]])
#         dataframe['j1_err'][path][p] = np.std(dataframe['j1'][path][p], ddof = 1)/np.sqrt(M) #not correct! need to use std of each angle average
#         std = np.average([np.std(angles, ddof = 1) for angles in dataframe['j2'][path][p]])
#         dataframe['j2_err'][path][p] = np.std(dataframe['j2'][path][p], ddof = 1)/np.sqrt(M)
#         std = np.average([np.std(angles, ddof = 1) for angles in dataframe['j3'][path][p]])
#         dataframe['j3_err'][path][p] = np.std(dataframe['j3'][path][p], ddof = 1)/np.sqrt(M)
#         dataframe['j1_sum'][path][p] = sum(dataframe['j1'][path][p])
#         dataframe['j2_sum'][path][p] = sum(dataframe['j2'][path][p])
#         dataframe['j3_sum'][path][p] = sum(dataframe['j3'][path][p])

#%%
# plot distance

col = iter(_col)
for path in path_type:
    colour = next(col)
    x = P
    y = [np.average(dataframe['d'][path][p]['raw']) for p in P]
    yerr = [np.std(dataframe['d'][path][p]['raw']) for p in P]
    plt.plot(x, y, color = colour)
    plt.errorbar(x, y, yerr = yerr, label = path, fmt = '.', ms = 20, capsize = 10, color = colour)

plt.xlabel('p')
plt.ylabel('geometric distance')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()
#%%
# plot average angle
angles = ['j1', 'j2', 'j3']
for ang in angles:
    col = iter(_col)
    for path in path_type:
        colour = next(col)
        x = P
        y = [np.average(dataframe[ang][path][p]['mean']) for p in P]
        yerr = [np.average(dataframe[ang][path][p]['err'])/np.sqrt(M) for p in P] # not entirely sure if this is the correct mathematics
        plt.plot(x, y, color = colour)
        plt.errorbar(x, y, yerr = yerr, label = path, fmt = '.', ms = 20, capsize = 10, color = colour)

    plt.xlabel('p')
    plt.ylabel(ang + 'average')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

#%%
# plot cumulative angle (looks relatively uninteresting so far, not sure if we need it)
angles = ['j1', 'j2', 'j3']
for ang in angles:
    col = iter(_col)
    for path in path_type:
        colour = next(col)
        x = P
        y = [np.average(dataframe[ang][path][p]['sum']) for p in P]
        yerr = [np.std(dataframe[ang][path][p]['sum']) for p in P] # not entirely sure if this is the correct mathematics
        plt.plot(x, y, color = colour)
        plt.errorbar(x, y, yerr = yerr, label = path, fmt = '.', ms = 20, capsize = 10, color = colour)
    plt.xlabel('p')
    plt.ylabel(ang + 'sum')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

#%%
# plot separation
separation = ['s1', 's2']
for s in separation:
    col = iter(_col)
    for path in path_type:
        colour = next(col)
        x = P
        y = [np.average(dataframe[s][path][p]['mean']) for p in P]
        yerr = [np.average(dataframe[s][path][p]['err'])/np.sqrt(M) for p in P]
        plt.errorbar(x, y, yerr = yerr, label = path, fmt = '.', ms = 20, capsize = 10, color = colour)
    plt.xlabel('p')
    plt.ylabel(s)
    plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    plt.show()
