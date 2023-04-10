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
    
from mpl_toolkits.axes_grid1 import make_axes_locatable
from DAG_Library.custom_functions import file_id
import matplotlib.pyplot as plt
import matplotlib as mpl
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
#%% PRINT PERCOLATION RESULTS 
for d in D:
    col = iter(['green', 'blue', 'red', 'm'])
    for p in [0.5, 1, 2]:
        # col = iter(['green', 'blue', 'red', 'm'])
        shapes = iter(['.', '^', '*', 'd'])
        colour = next(col)
        for rho in [2048, 4096]:
            #colour = next(col)
            shape = next(shapes)
            x = np.array([k for k in K[0::5] if k<4])
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            x_min = [k for k in K[0::5] if k<4].index(min([x[i] for i in range(len(x)) if y[i] > 0])) 
            x_max = [k for k in K[0::5] if k<4].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
            w, z, params, cov = funcfit(frechet, x[x_min:x_max], y[x_min:x_max], p0 = [2, -8], sigma = yerr[x_min:x_max])
            z0, z1 = params
            x_p = z0/((1/z1 + 1)**(1/z1))
            # plt.plot(x, y, label = rf'p={p}, $\rho$ = {rho}')
            #print(params)
            plt.ylabel(r'$\Pi(\langle k \rangle)$')
            plt.xlabel(r'$\langle k \rangle $')
            # plt.yscale('log')
            # plt.xscale('log')
            # plt.ylim(10e-2, 1)
            plt.xlim(x[x_min], x[x_max])
            plt.errorbar(x, y, yerr = yerr, fmt = shape, capsize = 4, ms = 8,
                         label = rf'p={p}, $\rho$ = {rho}', c = colour, markerfacecolor = 'none',
                         markeredgewidth = 1)
            plt.plot(w, z, c = colour, alpha = 0.5, linestyle = '--')
            plt.legend()
    plt.show()
# plt.ylabel(r'$\Pi(\langle k \rangle)$')
# plt.xlabel(r'$\langle k \rangle $')
# plt.legend()
# # plt.xlim(1.1, 3)
# plt.show()
#%%
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize = (12,12), gridspec_kw={'height_ratios': [2, 1]})
col = iter(['green', 'blue', 'red', 'm', 'k'])
#print(next(col))
shapes = iter(['.', 'd', '^', 's', '*'])
#for d in D:
for p in P[:]:
    rho = 5792
    colour = next(col)
    shape = next(shapes)
    x = np.array([k for k in K])
    y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
    yerr = [np.sqrt(M*y*(1-y))/M for y in y]
    x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 0])) 
    x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))

    # plt.plot(x, y, label = rf'p={p}, $\rho$ = {rho}')
    #print(params)
    ax1.set_ylabel(r'$\Pi(\langle k \rangle)$', fontsize = 28)
    ax1.set_xlabel(r'$\langle k \rangle $', fontsize = 28)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.ylim(10e-2, 1)
    ax1.set_xlim(x[x_min], x[x_max])
    ax1.errorbar(x, y, yerr = yerr, fmt = shape, capsize = 4, ms = 6, label = rf'p={p}, $\rho$ = {rho}', color = colour)
    #plt.plot(w, 1-z, c = colour, alpha = 0.7, linestyle = '--')
    ax1.legend()

#plt.savefig('ariel_figs\percolation-fig1.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
#plt.show()

#RHO = [512, 1024, 2048, 4096]
normalize = mpl.colors.Normalize(vmin=min(RHO), vmax=max(RHO))
cmap = mpl.cm.get_cmap('Dark2')
shapes = iter(['.', 'd', '^', 's'])
for d in D:
    for rho in RHO[5:]:
        p=2
        # colour = next(col)
        shape = next(shapes)
        x = np.array([k for k in K])
        y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
        yerr = [np.sqrt(M*y*(1-y))/M for y in y]
        x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 0])) 
        x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))

        # plt.plot(x, y, label = rf'p={p}, $\rho$ = {rho}')
        #print(params)
        ax2.set_ylabel(r'$\Pi(\langle k \rangle)$', fontsize = 28)
        ax2.set_xlabel(r'$\langle k \rangle $', fontsize = 28)
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.ylim(10e-2, 1)
        ax2.set_xlim(x[x_min], x[x_max])
        ax2.errorbar(x, y, yerr = yerr, fmt = shape, capsize = 3, ms = 5, label = rf'p={p}, $\rho$ = {rho}', color = cmap(normalize(rho)))
        #plt.plot(w, 1-z, c = colour, alpha = 0.7, linestyle = '--')
        ax2.legend()

#plt.savefig('ariel_figs\percolation-fig1.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.tight_layout()
plt.show()

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
            plt.plot(w, z, label = f'{rho}')
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
for x in np.arange(0.10, 0.25, 0.005):
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

#C = -np.log(Cost(kappa, x_range, p = 2))
C = Cost(kappa, x_range, p = 2)
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
    cmax = min([min(c) for c in C])
    kappamax, xmax= np.where(C == cmax)
    return kappa[kappamax[0]][xmax[0]], x_range[kappamax[0]][xmax[0]], cmax

# _i = 0
# for p in P:
#     _i += 0.05
#     kappa = [k for k in K if k > 1.8 + _i and k < 2.4 + _i]
#     x_range = np.arange(0.12, 0.24, 0.0005)
#     #C = np.power(1.1, np.log(Cost(kappa, x_range, p = p)))
#     C = np.log(Cost(kappa, x_range, p = p))
    
#     # C[C < 5] = 5
    
#     x_range, kappa = np.meshgrid(x_range, kappa)    
#     kappamax, xmax, cmax = maxCost(C)
#     print(kappamax, xmax, cost(kappamax, xmax, d=2, p=p))

#     c = plt.pcolormesh(kappa, x_range, C, cmap ='magma', shading = 'auto')
#     plt.scatter(kappamax, xmax, marker = '*', color = 'magenta', s =100)
    
#     plt.colorbar(c, label = 'Bastas Cost')
#     plt.xlabel(r'$\langle k \rangle$')
#     plt.ylabel(r'$\beta/\nu$')
#     plt.title(f'Bastas Cost Colormap for p={p}')
#     plt.show()
fig, axs = plt.subplots(3,2, figsize=[14,15])
axs = axs.flatten()
_i = 0
for p,ax in zip(P, axs):
    _i += 0.1
    kappa = [k for k in K if k > 1.8 + _i and k < 2.3 + _i]
    x_range = np.arange(0.12, 0.24, 0.0005)
    #C = np.power(1.1, np.log(Cost(kappa, x_range, p = p)))
    C = Cost(kappa, x_range, p = p)
    
    # C[C < 5] = 5
    ax.xaxis.set_ticks(np.arange(kappa[0], kappa[-1], 0.1))
    ax.yaxis.set_ticks(np.arange(0.12, 0.24, 0.025))
    x_range, kappa = np.meshgrid(x_range, kappa)    
    kappamax, xmax, cmax = maxCost(C)
    print(kappamax, xmax, cost(kappamax, xmax, d=2, p=p))

    c = ax.pcolormesh(kappa, x_range, C, cmap ='magma', shading = 'auto', norm = mpl.colors.LogNorm())
    ax.scatter(kappamax, xmax, marker = '*', color = 'cyan', s =100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '5%', pad = 0.1)
    fig.colorbar(c, label = 'Bastas Cost', cax = cax, orientation = 'vertical')
    #ax.colorbar(c, label = 'Bastas Cost')
    ax.set_xlabel(r'$\langle k \rangle$')
    ax.set_ylabel(r'$\beta/\nu$')
    ax.set_title(f'p={p}', fontsize = 16)

    
    ax.yaxis.get_ticklocs(minor = True)
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', length = 5)
    ax.tick_params(axis='y', which='major', length = 10)

    ax.xaxis.get_ticklocs(minor = True)
    ax.tick_params(axis='x', which='minor', length = 5)
    ax.tick_params(axis='x', which='major', length = 10)

fig.delaxes(axs[-1])
plt.tight_layout()
plt.savefig('clean_figs/bastas.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()
#%%
#empirical minimizer
# def mincost(dataframe, d, p, x0 = 0.33):
#     kappa = [k for k in K if k > 2 and k < 4]
#     # X = {rho: {k: [] for k in K if k > 2 and k < 4} for rho in RHO}
#     # for d in D:
#     #     for p in P:
#     #         for rho in RHO:
#     #             for k in kappa:
#     #                 X[rho][k] = dataframe[d][p][k][rho]['p']/M

#     x = x0
#     dx = 0.1 * x0
#     dc = np.inf
#     while abs(dc) > 0.00000000001:
#         c = min([cost(k, x, d, p) for k in kappa])
#         dc = min([cost(k, (x+dx), d, p) for k in kappa])  - c
#         if dc <0:
#             x = x + dx 
#             dx = dx/2
#         if dc >0:
#             dx = -dx/2
#         print(dx)
#     cost_min = min([cost(k, x, d, p) for k in kappa])
#     k_c = kappa[[cost(k, x, d, p) for k in kappa].index(cost_min)]
#     return k_c, x

# k_c, x = mincost(dataframe, d=2, p=2, x0 = 0.15)
# print(k_c, x, cost(k_c, x))
#%% TRADITIONAL EXPONENT PLOT
import random
r = lambda: random.randint(50,200)
print('#%02X%02X%02X' % (r(),r(),r()))

def funcfit(func, xdata, ydata, x = None, **kwargs):
    params, cov = op.curve_fit(func, xdata, ydata, **kwargs)
    if x == None:
        x = np.linspace(xdata[0], xdata[-1], 500)
        y = func(x, *params)
    else:
        y = func(x, *params)
    return x, y, params, cov

def power_law(x, a, delta):
    y = a*np.array(x)**delta
    return y



fig, ax = plt.subplots(1,1)
paramslist = []
errlist = []
for d in D:
    for p in P[3:4]:
        for k in [k for k in K if 2.1<k<2.56]:
            y = [dataframe[d][p][k][rho]['p']/M for rho in RHO if rho>1000]
            # if min(y)<0.4 or max(y)>0.6:
            #     continue
            x = [rho for rho in RHO if rho>1000]
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            col = '#%02X%02X%02X' % (r(),r(),r())
            w, z, params, cov = funcfit(power_law, x, y, sigma = yerr)
            exponent = params[1]
            exponent_std = np.sqrt(np.diag(cov))[1]
            if exponent_std < 0.01:
                paramslist.append([k, exponent])
                errlist.append(exponent_std)
                ax.errorbar(x,y,yerr = yerr, fmt = 'x', ms=10, capsize = 10, color = col)
                ax.plot(w, z, color = col, linestyle = '--', 
                         label = rf'$\langle k \rangle$ = {round(k, 3)}, $\beta/\nu$ = {round(-1*exponent, 3)} $\pm$ {round(exponent_std, 3)}')
                ax.set_yscale('log')
                ax.set_xscale('log')
                
        #set y ticks
        # y_major = mpl.ticker.LogLocator(base = 10, numticks = 10)
        # ax.yaxis.set_major_locator(y_major)
        # y_minor = mpl.ticker.LogLocator(base = 10, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
        # ax.yaxis.set_minor_locator(y_minor)
        # ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # ax.tick_params(axis='y', which='minor', length = 5)
        # ax.tick_params(axis='y', which='major', length = 10)
        # ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', length = 5)
        ax.tick_params(axis='y', which='major', length = 10)
        # #set x ticks
        # x_major = mpl.ticker.LogLocator(base = 10, numticks = 10)
        # ax.xaxis.set_major_locator(x_major)
        # x_minor = mpl.ticker.LogLocator(base = 10, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
        # ax.xaxis.set_minor_locator(x_minor)
        # ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.tick_params(axis='x', which='minor', length = 5)
        ax.tick_params(axis='x', which='major', length = 10)
        ax.legend(fontsize = 18)
        ax.set_ylabel(r'$\Pi(\langle k \rangle)$', fontsize = 25)
        ax.set_xlabel(r'$\rho$', fontsize = 25)
        plt.show()
            
#%% MINIMIZE BASTAS
from scipy.optimize import minimize
pmin = 4
kappa = [k for k in K if k > 2.18 and k < 2.6]
x_range = np.arange(0.12, 0.24, 0.000005)
def func(x, d=2, p=pmin):
    vals = Cost(kappa, x, d=d, p=p)
    #print(p)
    return min(vals) #vals.tolist().index(min(vals))#, kappa[C.index(min(C))]

x0 = 0.15

res = minimize(func, x0, tol = 1e-10)
print(res.x)
print(res.fun)

test = Cost(kappa, res.x[0], d=2, p=pmin)
mink = kappa[test.tolist().index(min(test))]
print(mink)
            
#%% PLOT ALL MINIMA FOR P=CONST
fig, (ax) = plt.subplots(1,1)

pmin = 2
x_range = np.arange(0.09, 0.22, 0.0005)
y = [func(x, d=2, p = pmin) for x in x_range]
ax.plot(x_range, y, label = r'$F(x)$,  min. cost surface')
oldkm, oldxm = paramslist[errlist.index(min(errlist))]
oldxm *= -1
newxm = x_range[y.index(min(y))]
#test = Cost(kappa, newxm, d=2, p=pmin)
#newkm = kappa[test.tolist().index(min(test))]

ax.scatter(oldxm, cost(oldkm, oldxm, p=pmin), marker = 'x', color = 'green', s =100, label = 'Traditional Method')
ax.scatter(newxm, min(y), marker = '*', color = 'magenta', s =200, label = 'New Method')
ax.legend()
ax.yaxis.get_ticklocs(minor = True)
ax.minorticks_on()
ax.tick_params(axis='y', which='minor', length = 5)
ax.tick_params(axis='y', which='major', length = 10)

ax.xaxis.get_ticklocs(minor = True)
ax.tick_params(axis='x', which='minor', length = 5)
ax.tick_params(axis='x', which='major', length = 10)
ax.set_ylabel(r'$Bastas \ Cost$', fontsize = 28)
ax.set_xlabel(r'$Critical \ exponent \ \beta/\nu$', fontsize = 28)
plt.savefig('clean_figs/bastas_local_minima.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()
#%%

C = np.zeros((3,len(x_range)))

for i in range(len(P[2:])):
    C[i] = [func(x, d=2, p = P[2:][i]) for x in x_range]
    plt.plot(x_range, C[i])
    
plt.show()
#%%  
y, x = np.meshgrid(x_range, P[2:])  
plt.pcolormesh(x, y, C, cmap ='viridis_r', shading = 'auto')
#plt.show()


