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

params = {
        'axes.labelsize':16,
        'axes.titlesize':28,
        'font.size':20,
        'figure.figsize': [11,11],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

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
fname2 = 'percolation_data_40000_2-3'
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
D = [2] #config[2]
K = config[3]
M = config[4]
P = config[5]

#%% PRINT PERCOLATION PLOT
for d in D:
    for p in P:
        for rho in RHO:
            x = [k for k in K]
            y = [dataframe[d][p][k][rho]['p']/M for k in K]
            plt.plot(x, y, label = rf'p={p}, $\rho$ = {rho}')
plt.ylabel(r'$\Pi(\langle k \rangle)$')
plt.xlabel(r'$\langle k \rangle $')
plt.legend()
plt.xlim(1.5, 3)
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
            plt.show()
            print(x_p[dydx.index(max(dydx))])
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
    sigma = 0
    for i in RHO:
        for j in RHO:
            if i != j:
                sigma += (H(i, k, x) - H(j, k, x))**2
    return sigma

def mincost(dataframe, x0 = 0.33):
    kappa = [k for k in K if k > 2 and k < 4]
    X = {rho: {k: [] for k in K if k > 2 and k < 4} for rho in RHO}
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

def Ye(rho, k, x):
    return Y(rho, k, x)

def gamma(rho, k, x, ye):
    return Y(rho, k, x)/ye

def He(rho, k, x, ye):
    He = gamma(rho, k, x, ye) + 1/gamma(rho, k, x, ye)
    return He    

def cost2(k ,x, ye):
    sigma = 0
    for rho in RHO:
        sigma += (He(rho, k, x, ye))#-2)
    return (sigma)
   
ye = Ye(1024, 2.2, 0.36)
he = He(2048, 2.2, 0.31, ye)
c = cost2(2.2, 0.36, ye)
x = 0.33
print(he, c)
kappa= [k for k in K if k > 2.2 and k < 2.7]
cs = [cost2(kap, x, Ye(2048, 2.3, 0.34)) for kap in kappa]
# plt.plot(kappa, cs)
# # print(min(cs))
# # plt.yscale('log')
# plt.show()

for x in np.arange(0.12, 0.18, 0.005):
    plt.plot(kappa, [cost(kap, x) for kap in kappa], label = 'x = %s' % (np.round(x,3)))
    plt.legend()
    plt.yscale('log')
    # print(x, min([cost2(kap, x, Ye(2048, kap, x)) for kap in kappa]))
plt.show()
     
#%%        

k_c, x = mincost(dataframe, x0 = 0.15)
print(k_c, x, cost(k_c, x))#, cost(2.3, 0.29))
for rho in RHO:
    plt.plot([k for k in K if k >2 and k < 2.5], [H(rho, k, x) for k in K if k >2 and k < 2.5], 'x')
    # plt.plot([k for k in K if k >2 and k < 2.5], [Y(rho, k, x) for k in K if k >2 and k < 2.5], 'x')
#cost(2.28, 0.27)
# cost(2.28, x)

    
#%% PRINT 3D PLOT WRT K, X
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

kappa = [k for k in K if k > 2.18 and k < 2.33]
x_range = np.arange(0.28, 0.32, 0.002)

def Cost(K, x):
    costs = []
    for k in K:
        costs.append(cost(k, x))
    return np.array(costs)
C = Cost(kappa, x_range)

x_range, kappa = np.meshgrid(x_range, kappa)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


# Plot the surface.
surf = ax.plot_surface(x_range, kappa, C, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(1, 4)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.view_init(30, -30)

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

