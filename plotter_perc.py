# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 16:42:31 2023

@author: ariel & kevin
"""
import os
import pickle
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
    BboxConnectorPatch
from DAG_Library.custom_functions import file_id
import matplotlib.pyplot as plt
import numpy as np

#%% DEFINE FILE ID
def file_id(name, pathfolder = None, pkl = True, directory = None):
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
    if pathfolder == None:
        pathfolder = 'path_data'
    else:
        pathfolder = pathfolder
    __file_name = f'{name}'
    _file_name = str(__file_name).replace(' ', '-').replace(',', '').replace('[', '-').replace(']','-').replace('.','-')
    file_name = os.path.join(directory, f'DAG_data_files\{pathfolder}', f'{_file_name}.pkl')
    return file_name

#%% RETRIEVE FILE
fname2 = 'percolation_data_prelim_06'
pathfolder = 'percolation_data'
try: 
    dataframe = pickle.load(open(f'{file_id(fname2, pathfolder = pathfolder)}', 'rb'))
except:
    raise ValueError('NO DATAFILE FOUND:')

#%% GET KEYS
datakeys = list(dataframe.keys())[-1]
config = dataframe[datakeys]['constants']
RHO = config[0]
V = config[1]
D = config[2]
K = config[3]
M = config[4]
P = config[5]

#%% PRINT PERCOLATION PLOT
for d in D:
    for p in P:
        for rho in RHO:
            x = [k for k in K]
            y = [dataframe[d][p][k][rho]['p']/M for k in K]
            plt.plot(x, y, label = f'p={p}, rho = {rho}')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(2.0, 2.3)
plt.legend()
plt.show()
#%% PRINT 1ST DERIVATIVE OF PERCOLATION PLOT
for d in D:
    for p in P:
        for rho in RHO:
            x = [k for k in K]
            y = [dataframe[d][p][k][rho]['p'] for k in K]
            dx = [x[i+1] - x[i] for i in range(len(x)-1)]
            dy = [y[i+1] - y[i] for i in range(len(y)-1)]
            dydx = [dy[i]/dx[i] for i in range(len(dx))]
            x_p = [x[i] + dx[i]/2 for i in range(len(dx))]
            plt.plot(x_p, dydx)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlim(0, 5)
            plt.show()

#%% PRINT BASTAS PLOT
X = {rho:{k:[] for k in K} for rho in RHO}
for d in D:
    for p in P:
        for rho in RHO:
            for k in K:
                X[rho][k] = dataframe[d][p][k][rho]['p']/M
            # plt.plot(x, y, label = f'p={p}, rho = {rho}')

def Y(rho, k, x):
    Y = X[rho][k] * rho ** x
    return Y

def H(rho, k, x):
    H = Y(rho, k, x) + 1/Y(rho, k, x)
    return H

def cost(k, x):
    gamma = 0
    for i in RHO:
        for j in RHO:
            if i != j:
                gamma += (H(i, k, x) - H(j, k, x))**2
    return gamma

def mincost(dataframe, x0 = 0.33):
    kappa = [k for k in K if k > 2 and k < 2.5]
    X = {rho: {k: [] for k in K if k > 2 and k < 2.5} for rho in RHO}
    for d in D:
        for p in P:
            for rho in RHO:
                for k in kappa:
                    X[rho][k] = dataframe[d][p][k][rho]['p']/M
    
    x = x0
    dx = 0.1 * x0
    dc = np.inf
    while abs(dc) > 0.00000000001:
        c = min([cost(k, x) for k in kappa])
        dc = min([cost(k, (x+dx)) for k in kappa])  - c
        if dc <0:
            x = x + dx 
            dx = dx/2
        if dc >0:
            dx = -dx/2
        print(dx)
    cost_min = min([cost(k, x) for k in kappa])
    k_c = kappa[[cost(k,x) for k in kappa].index(cost_min)]
    return k_c, x

k_c, x = mincost(dataframe)
print(k_c, x)
for rho in RHO:
    plt.plot([k for k in K if k >2 and k < 2.4], [H(rho, k, x) for k in K if k >2 and k < 2.4], 'x')