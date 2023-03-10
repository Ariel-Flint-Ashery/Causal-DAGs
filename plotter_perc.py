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
import scipy.optimize as op

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
fname2 = 'percolation_data_prelim_07'#percolation_data_prelim_06 #percolation_data_40000_2-3
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
D = config[2] #[2]
K = config[3]
M = config[4]
P = config[5]

#%% PRINT PERCOLATION PLOT
def empirical_derivative(x, y):
    dydx = []
    x_bar = []
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        dydx.append(dy/dx)
        x_bar.append((x[i+1] + x[i])/2)
    return np.array(x_bar), np.array(dydx)
def pareto(x, xm, alpha):
    return (1 - (xm/x)**alpha)
def pareto2(x, xm, alpha):
    return np.exp(-(xm/x)**alpha)
def dpareto2(x, xm, alpha):
    y = pareto2(x, xm, alpha)
    return empirical_derivative(x, y)

def funcfit(func, xdata, ydata, x = None, **kwargs):
    params, cov = op.curve_fit(func, xdata, ydata, **kwargs)
    if x == None:
        x = np.linspace(xdata[0], xdata[-1], 500)
        y = pareto(x, *params)
    else:
        y = pareto(x, *params)
    return x, y, params, cov

shapes = iter(['.', '.', '.'])
for d in D:
    for p in P[:1]:
        for rho in RHO:
            x = [k for k in K]
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            Y = y * rho ** 0.16
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            w, z, params, cov = funcfit(pareto, x[35:], y[35:])
            # plt.plot(x, y, label = rf'p={p}, $\rho$ = {rho}')
            print(params)
            plt.plot(w, z)
            plt.errorbar(x, y, yerr = yerr, fmt = '.', capsize = 3, ms = 1, label = rf'p={p}, $\rho$ = {rho}')
            plt.ylabel(r'$\Pi(\langle k \rangle)$')
            plt.xlabel(r'$\langle k \rangle $')
            plt.yscale('log')
            plt.xscale('log')
            plt.ylim(10e-2, 1)
            plt.xlim(1.5,6)
            plt.legend()
            plt.show()
# plt.ylabel(r'$\Pi(\langle k \rangle)$')
# plt.xlabel(r'$\langle k \rangle $')
# plt.legend()
# # plt.xlim(1.1, 3)
# plt.show()
#%% PRINT 1ST DERIVATIVE OF PERCOLATION PLOT
for d in D:
    for p in P[-1:]:
        for rho in RHO:
            x = [k for k in K]
            y = [dataframe[d][p][k][rho]['p'] for k in K]
            dx = [x[i+1] - x[i] for i in range(len(x)-1)]
            dy = [y[i+1] - y[i] for i in range(len(y)-1)]
            dydx = [dy[i]/dx[i] for i in range(len(dx))]
            x_p = [x[i] + dx[i]/2 for i in range(len(dx))]
            dydxerr = [d/np.sqrt(M) for d in dydx]
            # plt.plot(x_p, dydx, label = rf'p={p}, $\rho$ = {rho}')
            plt.errorbar(x_p, dydx, dydxerr, fmt = '.', capsize = 3, label = rf'p={p}, $\rho$ = {rho}')
            # plt.show()
            plt.legend()
            plt.ylabel(r'$\frac{d}{d\langle k \rangle} \Pi(\langle k \rangle)$')
            plt.xlabel(r'$\langle k \rangle$')
            # print(x_p[dydx.index(max(dydx))])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(1)
            plt.show()
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
#%% PRINT BASTAS PLOT

X = {d: {p:{rho:{k:[] for k in K} for rho in RHO} for p in P}for d in D}
for d in D:
    for p in P:
        for rho in RHO:
            for k in K:
                X[d][p][rho][k] = dataframe[d][p][k][rho]['p']/M
            # plt.plot(x, y, label = f'p={p}, rho = {rho}')

def Y(rho, k, x, d = 2, p = 2):
    Y = X[d][p][rho][k] * rho ** x
    return Y

def H(rho, k, x, d = 2, p = 2):
    H = Y(rho, k, x, d, p) + 1/Y(rho, k, x, d, p)
    return H

def cost(k, x, d = 2, p = 2):
    sigma = 0
    for i in RHO:
        for j in RHO:
            if i != j:
                sigma += 0.5 * (H(i, k, x, d, p) - H(j, k, x, d, p))**2
    return sigma

# def mincost(dataframe, x0 = 0.33):
#     kappa = [k for k in K if k > 2 and k < 4]
#     X = {rho: {k: [] for k in K if k > 2 and k < 4} for rho in RHO}
#     for d in D:
#         for p in P:
#             for rho in RHO:
#                 for k in kappa:
#                     X[rho][k] = dataframe[d][p][k][rho]['p']/M
    
#     x = x0
#     dx = 0.1 * x0
#     dc = np.inf
#     while abs(dc) > 0.00000000001:
#         c = min([cost(k, x) for k in kappa])
#         dc = min([cost(k, (x+dx)) for k in kappa])  - c
#         if dc <0:
#             x = x + dx 
#             dx = dx/2
#         if dc >0:
#             dx = -dx/2
#         print(dx)
#     cost_min = min([cost(k, x) for k in kappa])
#     k_c = kappa[[cost(k,x) for k in kappa].index(cost_min)]
#     return k_c, x

def Ye(rho, k, x, d = 2, p = 2):
    return Y(rho, k, x, d, p)

def gamma(rho, k, x, ye, d = 2, p = 2):
    return Y(rho, k, x, d, p)/ye

def He(rho, k, x, ye, d = 2, p = 2):
    He = gamma(rho, k, x, ye, d, p) + 1/gamma(rho, k, x, ye, d, p)
    return He    

def cost2(k ,x, d = 2, p = 2):     
    Yps = [Ye(rho, k, x, d, p) for rho in RHO]
    sigma = 0
    for ye in Yps:
        for rho in RHO:
            sigma += (He(rho, k, x, ye, d, p) - 2)
    return (sigma)

kappa = [k for k in K if k >2 and k < 2.7]
for x in np.arange(0.10, 0.25, 0.01):
    plt.plot(kappa, [cost(kap, x, d = 2, p = 0.5) for kap in kappa], label = np.round(x,3))
    plt.legend()
    plt.yscale('log')
plt.ylabel('Bastas Cost')
plt.xlabel('k')
plt.show()


kappa = [k for k in K if k >2 and k < 3]
for x in np.arange(0.1, 0.2, 0.005):
    plt.plot(kappa, [cost2(kap, x, p = 1) for kap in kappa], label = 'x = %s' % (np.round(x,3)))
    # plt.plot(kappa, [He(RHO[0], kap, x, Ye(RHO[4], kap, x, p = 2)) for kap in kappa])
    plt.legend()
    plt.yscale('log')
plt.ylabel('(H-2) Cost Function')
plt.xlabel('k')
plt.show()
    
#%% PRINT 3D PLOT WRT K, X
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

kappa = [k for k in K if k > 2.18 and k < 2.52]
x_range = np.arange(0.1, 0.3, 0.002)

def Cost(K, x, d = 2, p = 2):
    costs = []
    for k in K:
        costs.append(cost(k, x, d=d, p=p))
    return np.array(costs)
C = -np.log(Cost(kappa, x_range, p = 2))

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
ax.view_init(10, 60)

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
#%% PRINT HEATMAP OF BASTAS COST
for p in P:
    kappa = [k for k in K if k > 1.7 and k < 3]
    x_range = np.arange(0.1, 0.37, 0.005)
    lp = p
    C = -np.log(Cost(kappa, x_range, p = lp))
    
    x_range, kappa = np.meshgrid(x_range, kappa)
    c = plt.pcolormesh(kappa, x_range, C, cmap ='magma', shading = 'auto')
    plt.colorbar(c, label = 'Bastas Cost')
    plt.xlabel(r'$\langle k \rangle$')
    plt.ylabel(r'$\beta/\nu$')
    plt.title(f'Bastas Cost Colormap for p={lp}')
    plt.show()
