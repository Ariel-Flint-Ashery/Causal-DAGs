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
from scipy.optimize import minimize


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
D = config[2] 
K = config[3]
M = config[4]
P = config[5]

#%% PRINT PERCOLATION PLOT
def frechet(x, s, alpha):
    return np.exp(-((x)/s)**-alpha)

def funcfit(func, xdata, ydata, x = None, **kwargs):
    params, cov = op.curve_fit(func, xdata, ydata, **kwargs)
    if x == None:
        x = np.linspace(xdata[0], xdata[-1], 500)
        y = func(x, *params)
    else:
        y = func(x, *params)
    return x, y, params, cov

# def kstest(x, emp_cdf, theo_cdf, args, print_results = False):
#     """
#     Kolmogorov-Smirnov (KS) test: a goodness of fit test between an empirical CDF E(x) and a theoretical CDF T(x) of the variable x.
#     The KS statistic D: Supremum of |T(x) - E(x)| across all x:
#         D = sup_x |T(x) - E(x)|,
#     where the subscript n indicates the number of observations/samples.
#     A perfect fit between E(x) and T(x) would have a KS statistic D = 0.
#     The KS statistic D follows a Kolmogoriv distribution PDF in the limit as the sample size -> infty.
#     We can find the value of D that corresponds to a confidence level of alpha:
#         Pr(D <= D_alpha) = 1 - alpha 
#     """
#     N = len(x)
#     D = max([abs(emp_cdf[i] - theo_cdf(x[i], *args)) for i in range(N)])
#     crit_val = 1.36/np.sqrt(M)
#     if print_results == True:
#         print(f'Test statistic: {D}, critical_value: {crit_val}: D < crit_val: {D < crit_val} ')
#     return D, crit_val

def kstest(params, x, ecdf, tcdf, M, print_results = False):
    N = len(x)
    D = max([abs(ecdf[i] - tcdf(x[i], *params)) for i in range(N)])
    crit_val = 1.36/np.sqrt(M)
    if print_results == True:
        print(f'Test statistic: {D}, critical_value: {crit_val}: D < crit_val: {D < crit_val} ')
    return D, crit_val
    

def KSfit(params, x, ecdf, tcdf, M, print_results = False):
    kst = lambda x0: kstest((x0[0], x0[1]), x, ecdf, tcdf, M)[0]
    mks = minimize(kst, params)
    params = mks.x
    # print(params)
    return x, frechet(x, *params), params

shapes = iter(['.', '.', '.'])
for d in D:
    for p in P[:]:
        for rho in [512, 1024, 2048, 4096]:
            x = np.array([k for k in K])
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 5/M])) 
            x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
            w, z, params, cov = funcfit(frechet, x[x_min:x_max], y[x_min:x_max], p0 = [2, -8], sigma = yerr[x_min:x_max])
            z0, z1 = params
            x_p = z0/((1/z1 + 1)**(1/z1))
            # plt.plot(x, y, label = rf'p={p}, $\rho$ = {rho}')
            # print(params)
            print(np.average([dataframe[d][p][k][rho]['r'] for k in K]), max(y))
            plt.ylabel(r'$\Pi(\langle k \rangle)$')
            plt.xlabel(r'$\langle k \rangle $')
            # plt.yscale('log')
            # plt.xscale('log')
            # plt.ylim(10e-2, 1)
            # plt.xlim(x[x_min], x[x_max])
            plt.errorbar(x, y, yerr = yerr, fmt = '.', capsize = 3, ms = 1, label = rf'p={p}, $\rho$ = {rho}')
            plt.plot(w, z, label = 'WLS')
            # plt.show()
            
            # kstest(x, y, frechet, params)
            w, z, params = KSfit(params, x, y, frechet, M)
            plt.plot(w, z, label = 'KS')
            
            epsilon = np.sqrt(np.log(2/0.05)/(2*M))
            y_band_low = [i - epsilon for i in y]
            y_band_up = [i + epsilon for i in y]
            plt.fill_between(x, y_band_low, y_band_up, alpha = 0.3)
            plt.legend()
            plt.show()
            
            # plt.plot(x, frechet(x, xme, alphae))
# plt.ylabel(r'$\Pi(\langle k \rangle)$')
# plt.xlabel(r'$\langle k \rangle $')
# plt.legend()
# # plt.xlim(1.1, 3)
# plt.show()

#%% PRINT PERCOLATION PLOT
def frechet(x, s, alpha):
    return np.exp(-((x)/s)**-alpha)

def funcfit(func, xdata, ydata, x = None, **kwargs):
    params, cov = op.curve_fit(func, xdata, ydata, **kwargs)
    if x == None:
        x = np.linspace(xdata[0], xdata[-1], 500)
        y = func(x, *params)
    else:
        y = func(x, *params)
    return x, y, params, cov

def invfrechet(y, s, alpha):
    return np.exp((-np.log(-np.log(y)))/alpha + np.log(s))

for d in D:
    for p in P[-2:-1]:
        for rho in RHO[-2:-1]:
            x = np.array([k for k in K])
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            alpha = 0.05
            epsilon = np.sqrt(np.log(2/alpha)/(2*M))
            y_band_low = [i - epsilon for i in y]
            y_band_up = [i + epsilon for i in y]
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 0.05])) 
            x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
            w, z, params, cov = funcfit(frechet, x[x_min:x_max], y[x_min:x_max], p0 = [2, -8], sigma = yerr[x_min:x_max])
            xm, alpha = params
            xme = xm 
            alphae = alpha + -0
            # Plotting positions?
            qe = y[x_min:]
            qt = frechet(x[x_min:], *params)
            plt.plot(x[x_min:], invfrechet(qe, xme, alphae), '+')
            plt.plot([1,6], [1,6])
            plt.show()
            print(epsilon)
            plt.plot(x, y)
            plt.fill_between(x, y_band_low, y_band_up, alpha = 0.3)
            plt.plot(x, frechet(x, xme, alphae))
            plt.show()
#%%
def linear(x, a, b):
    return a*x + b

for d in D:
    for p in P[:]:
        for rho in RHO[:]:
            x = np.array([k for k in K])
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            xmin = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 0]))
            X = np.log(x[xmin:])
            Y = -np.log(-np.log(y[xmin:]))
            W, Z, (a,b), cov = funcfit(linear, X, Y)
            plt.plot(X, Y)
            plt.plot(W, Z)
            plt.show()
            plt.plot(x, y)
            plt.plot(x, frechet(x, np.exp(-b/a), a))
            plt.show()
#%% PRINT 1ST DERIVATIVE OF PERCOLATION PLOT
def dfunc(x, xm, alpha):
    return np.exp(-(xm/x)**alpha) * (alpha * xm ** alpha)/(x**(alpha+1))

def likelihood(k, count, xm, alpha):
    L = -np.sum([count[i] * np.log(dfunc(k[i], xm, alpha)) for i in range(len(k)) if count[i] != 0])
    return L
    

for d in D:
    for p in P[:]:
        for rho in RHO[-2:-1]:
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
            print(likelihood(x_p, dydx *M, *params))
            z0, z1 = params
            xp = z0/((1/z1 + 1)**(1/z1))
            plt.plot(x_p, dydx, label = rf'p={p}, $\rho$ = {rho}')
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
#%% LIKELIHOOD
import random
r = lambda: random.randint(50,200)
print('#%02X%02X%02X' % (r(),r(),r()))

def Likelihood(params):
    xm, alpha = params
    L = -np.sum([dydx[i] * np.log(dfunc(x_p[i], xm, alpha)) for i in range(len(x_p)) if dydx[i] != 0])
    return L

for d in D:
    for p in P[:]:
        for rho in RHO[-2:-1]:
            x = np.array([k for k in K])
            y = np.array([dataframe[d][p][k][rho]['p']/M for k in K])
            dx = [x[i+1] - x[i] for i in range(len(x)-1)]
            dy = [y[i+1] - y[i] for i in range(len(y)-1)]
            dydx = [(dy[i]/dx[i]) for i in range(len(dx))]
            x_p = [x[i] + dx[i]/2 for i in range(len(dx))]
            # dydxerr = [d/np.sqrt(d*M) for d in dydx]
            
            w, z, params, cov = funcfit(dfunc, x_p, dydx, p0 = [2, 9])
            z = [z[i] for i in range(len(w)) if w[i] != 0]
            w = [w[i] for i in range(len(w)) if w[i] != 0]
            xm = params[0]
            alpha = params[1]
            # plt.plot(w, z)
            mle = minimize(Likelihood, params)
            print(likelihood(x_p, np.array(dydx), *params), Likelihood(params), Likelihood(mle.x))
            dalpha = np.array([i/10 for i in range(-10, 10)])
            dxm = np.array([i/50 for i in range(-10, 10)])
            col = '#%02X%02X%02X' % (r(),r(),r())
            plt.plot(alpha + dalpha , [likelihood(x_p, dydx, xm, Alpha) for Alpha in alpha + dalpha], color = col)
            plt.plot(alpha, likelihood(x_p, dydx, xm, alpha), 'x', color = col, ms = 20)
            plt.plot(mle.x[1], Likelihood(mle.x), '+', color = col, ms = 20)
            plt.title('alpha')
            plt.show()
            plt.plot(xm + dxm, [likelihood(x_p, dydx, XM, alpha) for XM in xm + dxm], color = col)
            plt.plot(xm, likelihood(x_p, dydx, xm, alpha), 'x', color = col, ms = 20)
            plt.title('xm')
            plt.show()
            
#%%
for d in D:
    for p in P[:]:
        for rho in RHO[:]:
            x = np.array([k for k in K])
            y = np.array([dataframe[d][p][k][rho]['p']/M for k in K])
            dx = [x[i+1] - x[i] for i in range(len(x)-1)]
            dy = [y[i+1] - y[i] for i in range(len(y)-1)]
            dydx = [(dy[i]/dx[i]) for i in range(len(dx))]
            x_p = [x[i] + dx[i]/2 for i in range(len(dx))]
            # dydxerr = [d/np.sqrt(d*M) for d in dydx]
            
            mle = minimize(Likelihood, [2,6])
            params = mle.x
            
            w = np.arange(x[0], x[-1], 0.01)
            z = dfunc(w, *params)
            
            u, v, olsparams, olscov = funcfit(dfunc, x_p, dydx, p0 = [2, 9])
            s, t, olsparamscdf, olscovcdf = funcfit(frechet, x, y, p0 = [2,9])
            
            
            plt.plot(x_p, dydx, 'x')
            plt.plot(w, z, color = 'green')
            plt.plot(u, v, color = 'red')
            plt.show()
            
            # plt.plot(x, 1-y)
            # plt.plot(x, 1-frechet(x, *params), color = 'green')
            # plt.plot(x, 1-frechet(x, *olsparams), color = 'red')
            # # plt.plot(x, 1-frechet(x, *olsparamscdf), color = 'blue')
            # plt.xlim(1.5)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()
            
