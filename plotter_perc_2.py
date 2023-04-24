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
    kst = lambda x0: kstest((x0[0], x0[1]), x, ecdf, tcdf, M, print_results)[0]
    mks = minimize(kst, params)
    params = mks.x
    # print(params)
    return x, frechet(x, *params), params

shapes = iter(['.', '.', '.'])
for d in D:
    for p in P[3:4]:
        for rho in RHO[-5:]:
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
            w, z, params = KSfit(params, x, y, frechet, M, print_results = True)
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

# for d in D:
#     for p in P[-2:-1]:
#         for rho in RHO[-2:-1]:
#             x = np.array([k for k in K])
#             y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
#             alpha = 0.05
#             epsilon = np.sqrt(np.log(2/alpha)/(2*M))
#             y_band_low = [i - epsilon for i in y]
#             y_band_up = [i + epsilon for i in y]
#             yerr = [np.sqrt(M*y*(1-y))/M for y in y]
#             x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 0.05])) 
#             x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
#             w, z, params, cov = funcfit(frechet, x[x_min:x_max], y[x_min:x_max], p0 = [2, -8], sigma = yerr[x_min:x_max])
#             xm, alpha = params
#             xme = xm 
#             alphae = alpha + -0
#             # Plotting positions?
#             qe = y[x_min:]
#             qt = frechet(x[x_min:], *params)
#             plt.plot(x[x_min:], invfrechet(qe, xme, alphae), '+')
#             plt.plot([1,6], [1,6])
#             plt.show()
#             print(epsilon)
#             plt.plot(x, y)
#             plt.fill_between(x, y_band_low, y_band_up, alpha = 0.3)
#             plt.plot(x, frechet(x, xme, alphae))
#             plt.show()

#%% PRINT 1ST DERIVATIVE OF PERCOLATION PLOT
def dfunc(x, xm, alpha):
    """

    Parameters
    ----------
    x : array
        The range of the control parameter for the distribution.
    xm : float
        scale of the Frechet distribution.
    alpha : float
        shape of the Frechet distribution.

    Returns
    -------
    array
        returns Y values of the Frechet PDF.

    """
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
            print(likelihood(x_p, dydx , *params))
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
## Compare MLE (+) and non-linear WLS (X)
# import random
# r = lambda: random.randint(50,200)
# print('#%02X%02X%02X' % (r(),r(),r()))

# def Likelihood(params):
#     xm, alpha = params
#     L = -np.sum([dydx[i] * np.log(dfunc(x_p[i], xm, alpha)) for i in range(len(x_p)) if dydx[i] != 0])
#     return L

# for d in D:
#     for p in P[:]:
#         for rho in RHO[-2:-1]:
#             x = np.array([k for k in K])
#             y = np.array([dataframe[d][p][k][rho]['p']/M for k in K])
#             dx = [x[i+1] - x[i] for i in range(len(x)-1)]
#             dy = [y[i+1] - y[i] for i in range(len(y)-1)]
#             dydx = [(dy[i]/dx[i]) for i in range(len(dx))]
#             x_p = [x[i] + dx[i]/2 for i in range(len(dx))]
#             # dydxerr = [d/np.sqrt(d*M) for d in dydx]
            
#             w, z, params, cov = funcfit(dfunc, x_p, dydx, p0 = [2, 9])
#             z = [z[i] for i in range(len(w)) if w[i] != 0]
#             w = [w[i] for i in range(len(w)) if w[i] != 0]
#             xm = params[0]
#             alpha = params[1]
#             # plt.plot(w, z)
#             mle = minimize(Likelihood, params)
#             #
#             print(f'WLS: {Likelihood(params)}, MLE: {Likelihood(mle.x)}') #likelihood(x_p, np.array(dydx), *params),
#             dalpha = np.array([i/10 for i in range(-10, 10)])
#             dxm = np.array([i/50 for i in range(-10, 10)])
#             col = '#%02X%02X%02X' % (r(),r(),r())
#             plt.plot(alpha + dalpha , [likelihood(x_p, dydx, xm, Alpha) for Alpha in alpha + dalpha], color = col)
#             plt.plot(alpha, likelihood(x_p, dydx, xm, alpha), 'x', color = col, ms = 20)
#             plt.plot(mle.x[1], Likelihood(mle.x), '+', color = col, ms = 20)
#             plt.title('alpha')
#             plt.show()
#             plt.plot(xm + dxm, [likelihood(x_p, dydx, XM, alpha) for XM in xm + dxm], color = col)
#             plt.plot(xm, likelihood(x_p, dydx, xm, alpha), 'x', color = col, ms = 20)
#             plt.plot(mle.x[0], Likelihood(mle.x), '+', color = col, ms = 20)
#             plt.title('xm')
#             plt.show()
            
#%%
# for d in D:
#     for p in P[:]:
#         for rho in RHO[:]:
#             x = np.array([k for k in K])
#             y = np.array([dataframe[d][p][k][rho]['p']/M for k in K])
#             dx = [x[i+1] - x[i] for i in range(len(x)-1)]
#             dy = [y[i+1] - y[i] for i in range(len(y)-1)]
#             dydx = [(dy[i]/dx[i]) for i in range(len(dx))]
#             x_p = [x[i] + dx[i]/2 for i in range(len(dx))]
#             # dydxerr = [d/np.sqrt(d*M) for d in dydx]
            
#             mle = minimize(Likelihood, [2,6])
#             params = mle.x
            
#             w = np.arange(x[0], x[-1], 0.01)
#             z = dfunc(w, *params)
            
#             u, v, olsparams, olscov = funcfit(dfunc, x_p, dydx, p0 = [2, 9])
#             s, t, olsparamscdf, olscovcdf = funcfit(frechet, x, y, p0 = [2,9])
            
            
#             plt.plot(x_p, dydx, 'x')
#             plt.plot(w, z, color = 'green')
#             plt.plot(u, v, color = 'red')
#             plt.show()
            
            # plt.plot(x, 1-y)
            # plt.plot(x, 1-frechet(x, *params), color = 'green')
            # plt.plot(x, 1-frechet(x, *olsparams), color = 'red')
            # # plt.plot(x, 1-frechet(x, *olsparamscdf), color = 'blue')
            # plt.xlim(1.5)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()
            

#%% PLOT FINAL FIGURES -- RHO = 4096, VARY P
fig, (ax1, ax2) = plt.subplots(2, 1,figsize = (15,15), gridspec_kw={'height_ratios': [2, 1]})

col = iter(['green', 'blue', 'red', 'm', 'k'])
shapes = iter(['.', 'd', '^', 's', '*'])
for d in D:
    for p in P[:]:
        rho = 4096
        colour = next(col)
        shape = next(shapes)
        x = np.array([k for k in K])
        y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
        yerr = [np.sqrt(M*y*(1-y))/M for y in y]
        x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 5/M])) 
        x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
        w, z, params, cov = funcfit(frechet, x[x_min:x_max], y[x_min:x_max], p0 = [2, -8], sigma = yerr[x_min:x_max])

        # kstest(x, y, frechet, params)
        w, z, params = KSfit(params, x, y, frechet, M, print_results = True)
        ax1.plot(w, z, linestyle = '--', alpha = 0.7, color = colour)
        
        ax1.set_ylabel(r'$\Pi(\langle k \rangle)$', fontsize = 28)
        ax1.set_xlabel(r'$\langle k \rangle $', fontsize = 28)
        
        x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.9]))
        ax1.set_xlim(x[x_min], x[x_max])
        
        ax1.errorbar(x, y, yerr = yerr, fmt = shape, capsize = 4, ms = 6, label = rf'p={p}', color = colour)
        #ax1.plot(w, 1-z, c = colour, alpha = 0.7, linestyle = '--') #PDF
        ax1.legend(loc = 'lower right', ncol = 3, fontsize = 22)
        ax1.yaxis.get_ticklocs(minor = True)
        ax1.minorticks_on()
        ax1.tick_params(axis='y', which='minor', length = 5)
        ax1.tick_params(axis='y', which='major', length = 10)

        ax1.xaxis.get_ticklocs(minor = True)
        ax1.tick_params(axis='x', which='minor', length = 5)
        ax1.tick_params(axis='x', which='major', length = 10)
        
        #plot fractional errors
        y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x[x_min:x_max]])/z[x_min:x_max]
        yerr = yerr[x_min:x_max]/z[x_min:x_max]
        ax2.axhline(1, linestyle = 'dotted', color = 'k')
        ax2.errorbar(x[x_min:x_max], y, yerr = yerr, fmt = shape, capsize = 4, ms = 6, label = rf'p={p}', color = colour)
        
        ax2.yaxis.get_ticklocs(minor = True)
        ax2.minorticks_on()
        ax2.tick_params(axis='y', which='minor', length = 5)
        ax2.tick_params(axis='y', which='major', length = 10)

        ax2.xaxis.get_ticklocs(minor = True)
        ax2.tick_params(axis='x', which='minor', length = 5)
        ax2.tick_params(axis='x', which='major', length = 10)
        
        ax2.set_ylabel(r'$\Pi(\langle k \rangle)/Fit$', fontsize = 28)
        ax2.set_xlabel(r'$\langle k \rangle $', fontsize = 28)
        ax2.set_xlim(x[x_min]+0.005, x[x_max])
        ax2.set_ylim(0.5, 1.2)
        ax2.legend(ncol=3, fontsize = 22)
        
plt.tight_layout()
plt.show()

#%% PLOT FINAL FIGURES -- VARY RHO, P=2
fig, (ax1, ax2) = plt.subplots(2, 1,figsize = (15,15), gridspec_kw={'height_ratios': [2, 1]})

col = iter(['green', 'blue', 'red', 'm', 'k'])
shapes = iter(['.', 'd', '^', 's', '*'])
for d in D:
    for p in P[3:4]: #choose constant P
        for rho in [1448, 2048, 4096]:
            colour = next(col) 
            shape = next(shapes)
            x = np.array([k for k in K])
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 5/M])) 
            x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
            w, z, params, cov = funcfit(frechet, x[x_min:x_max], y[x_min:x_max], p0 = [2, -8], sigma = yerr[x_min:x_max])
    
            # kstest(x, y, frechet, params)
            w, z, params = KSfit(params, x, y, frechet, M, print_results = True)
            ax1.plot(w, z, linestyle = '--', alpha = 0.7, color = colour)
            
            ax1.set_ylabel(r'$\Pi(\langle k \rangle)$', fontsize = 28)
            ax1.set_xlabel(r'$\langle k \rangle $', fontsize = 28)
            
            x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.9]))
            ax1.set_xlim(x[x_min], x[x_max])
            
            ax1.errorbar(x, y, yerr = yerr, fmt = shape, capsize = 4, ms = 6, label = rf'$\rho={rho}$', color = colour)
            #ax1.plot(w, 1-z, c = colour, alpha = 0.7, linestyle = '--') #PDF
            ax1.legend(loc = 'lower right', ncol = 3, fontsize = 24)
            ax1.yaxis.get_ticklocs(minor = True)
            ax1.minorticks_on()
            ax1.tick_params(axis='y', which='minor', length = 5)
            ax1.tick_params(axis='y', which='major', length = 10)
    
            ax1.xaxis.get_ticklocs(minor = True)
            ax1.tick_params(axis='x', which='minor', length = 5)
            ax1.tick_params(axis='x', which='major', length = 10)
            
            #plot fractional errors
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x[x_min:x_max]])/z[x_min:x_max]
            yerr = yerr[x_min:x_max]/z[x_min:x_max]
            ax2.axhline(1, linestyle = 'dotted', color = 'k')
            ax2.errorbar(x[x_min:x_max], y, yerr = yerr, fmt = shape, capsize = 4, ms = 6, label = rf'$\rho={rho}$', color = colour)
            
            ax2.yaxis.get_ticklocs(minor = True)
            ax2.minorticks_on()
            ax2.tick_params(axis='y', which='minor', length = 5)
            ax2.tick_params(axis='y', which='major', length = 10)
    
            ax2.xaxis.get_ticklocs(minor = True)
            ax2.tick_params(axis='x', which='minor', length = 5)
            ax2.tick_params(axis='x', which='major', length = 10)
            
            ax2.set_ylabel(r'$\Pi(\langle k \rangle)/Fit$', fontsize = 28)
            ax2.set_xlabel(r'$\langle k \rangle $', fontsize = 28)
            ax2.set_xlim(x[x_min]+0.005, x[x_max])
            ax2.set_ylim(0.4, 1.8)
            ax2.legend(ncol=3, fontsize = 24)

#annotate ax1

ax1.annotate("", xy=(2.2, 0.91), xycoords = 'data', 
            xytext=(2.55, 0.91), textcoords = 'data', 
            arrowprops=dict(width = 2, ec='k', fc = 'k', headwidth = 15, headlength = 25), color = 'k')

ax1.annotate("", xy=(3.275, 0.91), xycoords = 'data', 
            xytext=(2.925, 0.91), textcoords = 'data', 
            arrowprops=dict(width = 2, ec='k', fc = 'k', headwidth = 15, headlength = 25), color = 'k')

ax1.annotate(r'BROADENING', xy = (2.6, 0.9), xycoords = 'data',
             xytext = (2.6, 0.9), textcoords = 'data', fontsize = 22)

plt.tight_layout()
plt.show()

#%% FIND DERIVATIVES

fig, (ax) = plt.subplots(1,1)
col = iter(['green', 'blue', 'red', 'm', 'k'])
shapes = iter(['.', 'd', '^', 's', '*'])
for d in D:
    for p in P[:]:
        rho = 4096
        colour = next(col)
        shape = next(shapes)
        x = np.array([k for k in K])
        y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
        yerr = [np.sqrt(M*y*(1-y))/M for y in y]
        x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 5/M])) 
        x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
        w, z, params, cov = funcfit(frechet, x[x_min:x_max], y[x_min:x_max], p0 = [2, -8], sigma = yerr[x_min:x_max])

        # kstest(x, y, frechet, params)
        w, z, params = KSfit(params, x, y, frechet, M, print_results = True)
        x = np.linspace(1, x[x_max], 1000)
        y = dfunc(x, *params)
        s, alpha = params
        mode = s*(alpha/(alpha+1))**(1/alpha)
        ax.plot([mode, mode], [0, max(y)], linestyle = 
                '--', color = colour, alpha = 0.7)
        ax.plot(x, y, color = colour, label = rf'$p={p}, k_c = {round(mode, 3)}$')
        ax.set_ylabel(r'$\frac{d}{d\langle k \rangle} \Pi(\langle k \rangle)$', fontsize = 28)
        ax.set_xlabel(r'$\langle k \rangle$', fontsize = 28)
        
        ax.legend(loc = 'upper right', fontsize = 22)
        ax.yaxis.get_ticklocs(minor = True)
        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', length = 5)
        ax.tick_params(axis='y', which='major', length = 10)

        ax.xaxis.get_ticklocs(minor = True)
        ax.tick_params(axis='x', which='minor', length = 5)
        ax.tick_params(axis='x', which='major', length = 10)
                
        ax.set_xlim(1.96, 3.68)
        ax.set_ylim(0, 1.2)
        
# ax.annotate("", xy=(4.6, 0.4), xycoords = 'data', 
#             xytext=(3.7, 0.4), textcoords = 'data', 
#             arrowprops=dict(width = 2, ec='k', fc = 'k', headwidth = 15, headlength = 25), color = 'k')

# ax.annotate(r'$p$', xy = (4.1, 0.435), xycoords = 'data',
#              xytext = (4.1, 0.435), textcoords = 'data', fontsize = 26)

ax.annotate("", xy=(3.6, 0.565), xycoords = 'data', 
            xytext=(3.25, 0.565), textcoords = 'data', 
            arrowprops=dict(width = 2, ec='k', fc = 'k', headwidth = 15, headlength = 25), color = 'k')

ax.annotate(r'$p$', xy = (3.4, 0.6), xycoords = 'data',
             xytext = (3.4, 0.6), textcoords = 'data', fontsize = 26)

plt.tight_layout()
plt.show()

#%%
fig, (ax) = plt.subplots(1,1)
col = iter(['green', 'blue', 'red', 'm', 'k'])
shapes = iter(['.', 'd', '^', 's', '*'])
for d in D:
    for p in P[3:4]: #choose constant P
        for rho in [1448, 2048, 4096]:
            colour = next(col) 
            shape = next(shapes)
            x = np.array([k for k in K])
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 5/M])) 
            x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
            w, z, params, cov = funcfit(frechet, x[x_min:x_max], y[x_min:x_max], p0 = [2, -8], sigma = yerr[x_min:x_max])
    
            # kstest(x, y, frechet, params)
            w, z, params = KSfit(params, x, y, frechet, M, print_results = True)
            x = np.linspace(1, x[x_max], 1000)
            y = dfunc(x, *params)
            s, alpha = params
            mode = s*(alpha/(alpha+1))**(1/alpha)
            ax.plot([mode, mode], [0, max(y)], linestyle = 
                    '--', color = colour, alpha = 0.7)
            ax.plot(x, y, color = colour, label = rf'$\rho={rho}, k_c = {round(mode, 3)}$')
            ax.set_ylabel(r'$\frac{d}{d\langle k \rangle} \Pi(\langle k \rangle)$', fontsize = 28)
            ax.set_xlabel(r'$\langle k \rangle$', fontsize = 28)
            
            ax.legend(loc = 'upper right', fontsize = 22)
            ax.yaxis.get_ticklocs(minor = True)
            ax.minorticks_on()
            ax.tick_params(axis='y', which='minor', length = 5)
            ax.tick_params(axis='y', which='major', length = 10)
        
            ax.xaxis.get_ticklocs(minor = True)
            ax.tick_params(axis='x', which='minor', length = 5)
            ax.tick_params(axis='x', which='major', length = 10)
                    
            ax.set_xlim(1.92, 3.6)
            ax.set_ylim(0, 1.1)        

ax.annotate("", xy=(3.5, 0.665), xycoords = 'data', 
            xytext=(3.1, 0.665), textcoords = 'data', 
            arrowprops=dict(width = 2, ec='k', fc = 'k', headwidth = 15, headlength = 25), color = 'k')

ax.annotate(r'$\rho$', xy = (3.3, 0.7), xycoords = 'data',
             xytext = (3.3, 0.7), textcoords = 'data', fontsize = 26)

plt.tight_layout()
plt.show()

#%%
def Frechet(x, s, alpha):
    return np.exp(-((x)/s)**-alpha)

def Gumbel(x, mu, beta):
    return np.exp(-np.exp(-(x-mu)/beta))

def FrechetGumbel(x, s, alpha, mu, beta):
    return Frechet(x, s, alpha) * Gumbel(x, mu, beta)

def funcfit(func, xdata, ydata, x = None, **kwargs):
    params, cov = op.curve_fit(func, xdata, ydata, **kwargs)
    if x == None:
        x = np.linspace(xdata[0], xdata[-1], 500)
        y = func(x, *params)
    else:
        y = func(x, *params)
    return x, y, params, cov

def Residuals(func, params, xdata, ydata):
    xfit = xdata
    yfit = func(xfit, *params)
    res = ydata - yfit
    return res

def normResiduals(func, params, xdata, ydata, sigdata):
    xfit = xdata
    yfit = func(xfit, *params)
    nres = (ydata - yfit)/sigdata
    return nres

def reducedChiSq(func, params, xdata, ydata, sigdata):
    xfit = xdata
    yfit = func(xfit, *params)
    chisq = np.sum(((ydata - yfit)**2)/(np.array(sigdata)**2))
    ddof = len(xdata) - len(params)
    return chisq/ddof

shapes = iter(['.', '.', '.'])
params_dict = {p:{rho:{'params': None} for rho in RHO} for p in P}
for d in D:
    for p in P[:]:
        for rho in RHO: #[4096]:
            x = np.array([k for k in K])
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 0.01])) 
            x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
            
            X = x[x_min:x_max]
            Y = y[x_min:x_max]
            YERR = yerr[x_min:x_max]
            
            w, z, params0, cov = funcfit(Frechet, X, Y, p0 = [2, 8], sigma = YERR)
            # z0, z1 = params
            # x_p = z0/((1/z1 + 1)**(1/z1))
            s, t, params, cov = funcfit(FrechetGumbel, X, Y, sigma = YERR, p0 = [2, 8, 2, 2])
            params_dict[p][rho]['params'] = params
            # print(params, params[0])
            plt.ylabel(r'$\Pi(\langle k \rangle)$')
            plt.xlabel(r'$\langle k \rangle $')
            # plt.yscale('log')
            # plt.xscale('log')
            # plt.ylim(10e-2, 1)
            plt.xlim(x[x_min], x[x_max])
            plt.errorbar(x, y, yerr = yerr, fmt = '.', capsize = 3, ms = 1, label = rf'p={p}, $\rho$ = {rho}')
            plt.plot(w, z, label = 'frechet')
            # plt.plot(x, Frechet(x, *params[:2]))
            # plt.plot(x, Gumbel(x, *params[2:]))
            plt.plot(s, t, label = 'frechet * gumbel')
            epsilon = np.sqrt(np.log(2/0.1)/(2*M))
            y_band_low = [i - epsilon for i in y]
            y_band_low = [max([0,i]) for i in y_band_low]
            y_band_up = [i + epsilon for i in y]
            y_band_up = [min([1, i]) for i in y_band_up]
            # plt.fill_between(x, y_band_low, y_band_up, alpha = 0.3)
            plt.legend()
            plt.show()
            

            rchisq = reducedChiSq(FrechetGumbel, params, X, Y, YERR)
            print(rchisq)
            plt.show()
#%%

def frechet(x, s, alpha):
    return (alpha/s) * ((x/s) ** (-1-alpha)) * Frechet(x, s, alpha)

def gumbel(x, mu, beta):
    return (1/beta) * np.exp(-(x-mu)/beta) * Gumbel(x, mu, beta)
    
def frechetgumbel(x, s, alpha, mu, beta):
    return Frechet(x, s, alpha) * gumbel(x, mu, beta) + frechet(x, s, alpha) * Gumbel(x, mu, beta)

for d in D:
    for p in P[:]:
        for rho in [4096]:
            x = np.array([k for k in K])
            y = np.array([(dataframe[d][p][k][rho]['p']/M) for k in x])
            yerr = [np.sqrt(M*y*(1-y))/M for y in y]
            x_min = [k for k in K].index(min([x[i] for i in range(len(x)) if y[i] > 0])) 
            x_max = [k for k in K].index(max([x[i] for i in range(len(x)) if y[i] <0.99]))
            params = params_dict[p][rho]['params']
            fparams = params[:2]
            gparams = params[2:]
            
            u = np.linspace(x[0], x[-1], 1000)
            v = [i for i in frechetgumbel(u, *params)]
            plt.plot(u, v, label = f'p = {p}')
            plt.legend()
            plt.ylabel(rf'$\partial_x \; \Pi(k)$')
            plt.xlabel('k')
            print(u[v.index(max(v))])