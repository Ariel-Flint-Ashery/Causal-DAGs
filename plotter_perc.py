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
import scipy.special as sps
import scipy.integrate as integrate

params = {
        'axes.labelsize':16,
        'axes.titlesize':28,
        'font.size':20,
        'figure.figsize': [15,11],
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
fname2 = 'percolation_data_local_01' #percolation_data_prelim_06 #percolation_data_40000_2-3
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
def frechet(x, s, alpha):
    return np.exp(-((x)/s)**-alpha)

def gumbel(x, mu, sigma):
    return np.exp(-np.exp((x - mu)/sigma))

def funcfit(func, xdata, ydata, x = None, **kwargs):
    params, cov = op.curve_fit(func, xdata, ydata, **kwargs)
    if x == None:
        x = np.linspace(xdata[0], xdata[-1], 500)
        y = func(x, *params)
    else:
        y = func(x, *params)
    return x, y, params, cov

shapes = iter(['.', '.', '.'])
for d in D:
    for p in P[:]:
        for rho in [512, 1024, 2048, 4096]:
            x = np.array([k for k in K])
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 0])) 
            x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
            w, z, params, cov = funcfit(frechet, x[x_min:x_max], y[x_min:x_max], p0 = [2, -8], sigma = yerr[x_min:x_max])
            z0, z1 = params
            x_p = z0/((1/z1 + 1)**(1/z1))
            # plt.plot(x, y, label = rf'p={p}, $\rho$ = {rho}')
            print(params)
            plt.ylabel(r'$\Pi(\langle k \rangle)$')
            plt.xlabel(r'$\langle k \rangle $')
            # plt.yscale('log')
            # plt.xscale('log')
            # plt.ylim(10e-2, 1)
            plt.xlim(x[x_min], x[x_max])
            plt.errorbar(x, y, yerr = yerr, fmt = '.', capsize = 3, ms = 1, label = rf'p={p}, $\rho$ = {rho}')
            plt.plot(w, z)
            plt.legend()
        plt.show()
# plt.ylabel(r'$\Pi(\langle k \rangle)$')
# plt.xlabel(r'$\langle k \rangle $')
# plt.legend()
# # plt.xlim(1.1, 3)
# plt.show()
#%% PRINT 1ST DERIVATIVE OF PERCOLATION PLOT
def dfunc(x, xm, alpha):
    return np.exp(-(xm/x)**alpha) * (alpha * xm ** alpha)/(x**(alpha+1))

for d in D:
    for p in P[:]:
        for rho in RHO:
            x = np.array([k for k in K])
            y = np.array([dataframe[d][p][k][rho]['p']/M for k in K])
            dx = [x[i+1] - x[i] for i in range(len(x)-1)]
            dy = [y[i+1] - y[i] for i in range(len(y)-1)]
            dydx = [(dy[i]/dx[i]) for i in range(len(dx))]
            x_p = [x[i] + dx[i]/2 for i in range(len(dx))]
            dydxerr = [d/np.sqrt(d*M) for d in dydx]
            
            w, z, params, cov = funcfit(dfunc, x_p, dydx, p0 = [2, 9])
            z = [z[i] for i in range(len(w)) if w[i] != 0]
            w = [w[i] for i in range(len(w)) if w[i] != 0]
            plt.plot(w, z)
            z0, z1 = params
            xp = z0/((1/z1 + 1)**(1/z1))
            # plt.plot(x_p, dydx, label = rf'p={p}, $\rho$ = {rho}')
            # print(params, xp)
            # plt.errorbar(x_p, dydx, dydxerr, fmt = '.', capsize = 3, label = rf'p={p}, $\rho$ = {rho}')
            # plt.show()
            plt.ylabel(r'$\frac{d}{d\langle k \rangle} \Pi(\langle k \rangle)$')
            plt.xlabel(r'$\langle k \rangle$')
            # print(x_p[dydx.index(max(dydx))])
            # plt.xscale('log')
            # plt.yscale('log')
            plt.xlim(1)
            plt.ylim(0, 1.5)
            # plt.axvline(params[0], label = 'mean?')
            # plt.axvline(params[0]/((1+ 1/(params[1]))**(1/params[1])), color = 'red', label = 'mode')
            plt.legend()
            # plt.show()
        # plt.xscale('log')
        # plt.yscale('log')
        plt.show()
        
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

kappa = [k for k in K if k >1.8 and k < 2.7]
for x in np.arange(0.10, 0.25, 0.01):
    plt.plot(kappa, [cost(kap, x, d = 2, p = 0.5) for kap in kappa], label = np.round(x,3))
    plt.legend()
    plt.yscale('log')
plt.ylabel('Bastas Cost')
plt.xlabel('k')
plt.show()


kappa = [k for k in K if k >1.8 and k < 2.7]
for x in np.arange(0.1, 0.2, 0.005):
    plt.plot(kappa, [cost2(kap, x, p = 0.5) for kap in kappa], label = 'x = %s' % (np.round(x,3)))
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

# x_range, kappa = np.meshgrid(x_range, kappa)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


# # Plot the surface.
# surf = ax.plot_surface(x_range, kappa, C, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)

# # # Customize the z axis.
# # ax.set_zlim(1, 4)
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.view_init(10, 60)

# # # Add a color bar which maps values to colors.
# # fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
#%% PRINT HEATMAP OF BASTAS COST
def maxCost(C):
    cmax = max([max(c) for c in C])
    kappamax, xmax= np.where(C == cmax)
    return kappa[kappamax[0]][xmax[0]], x_range[kappamax[0]][xmax[0]], cmax

_i = 0
for p in P:
    _i += 0.05
    kappa = [k for k in K if k > 1.8 + _i and k < 2.5 + _i]
    x_range = np.arange(0.16, 0.24, 0.005)
    C = np.power(1.1, -np.log(Cost(kappa, x_range, p = p)))
    # C[C < 5] = 5
    
    x_range, kappa = np.meshgrid(x_range, kappa)    
    kappamax, xmax, cmax = maxCost(C)
    print(kappamax, xmax)

    c = plt.pcolormesh(kappa, x_range, C, cmap ='magma', shading = 'auto')
    plt.scatter(kappamax, xmax, marker = '*', color = 'magenta', s =100)
    
    plt.colorbar(c, label = 'Bastas Cost')
    plt.xlabel(r'$\langle k \rangle$')
    plt.ylabel(r'$\beta/\nu$')
    plt.title(f'Bastas Cost Colormap for p={p}')
    plt.show()
