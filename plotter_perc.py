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
fname2 = 'percolation_data_prelim_03'
pathfolder = 'percolation_data'
try: 
    dataframe = pickle.load(open(f'{file_id(fname2, pathfolder = pathfolder)}', 'rb'))
except:
    raise ValueError('NO DATAFILE FOUND:')

#%% GET KEYS
_, datakeys = dataframe.keys()
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
            y = [dataframe[d][p][k][rho]['p'] for k in K]
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
            plt.show()
