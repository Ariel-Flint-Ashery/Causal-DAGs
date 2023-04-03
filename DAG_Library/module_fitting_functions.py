# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 01:21:22 2023

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.optmize import minimize

def Dfunc(x, a, b):
    return 2 ** (1 - b**(-a) + (b*x)**(-a))

def Dfit(x, y, **kwargs):
    u = np.linspace(x[0], x[-1], 10000, endpoint = True)
    params, cov = op.curve_fit(Dfunc, x, y, **kwargs)
    v = Dfunc(u, *params)
    return u, v, params, cov

def swapdata(dataframe, measure = 'd'):
    path_type = list(dataframe['d'].keys())
    for path in path_type:
        if 'sp' in path:
            sp = path
        elif 'lp' in path:
            lp = path
        else:
            continue
    paths = [sp, lp]
    lscale = ['diagonal', 'perimeter']
    dffit = {'diagonal':{}, 'perimeter':{}}
    for i in [0, -1]:
        P = list(dataframe[measure][path].keys())
        y = [np.average(dataframe[measure][paths[i]][p]['raw']) for p in P if p<=1] + [np.average(dataframe[measure][paths[i+1]][p]['raw']) for p in P if p>1]
        yerr = [np.std(dataframe[measure][paths[i]][p]['raw']) for p in P if p<=1] + [np.std(dataframe[measure][paths[i+1]][p]['raw']) for p in P if p > 1]
        
        L = lscale[i]
        dffit[L]['p'] = P
        dffit[L][measure] = y
        dffit[L][measure + '_err'] = yerr
    
    return dffit

def Frechet(x, s, alpha):
    """
    Cumulative distribution function of the Frechet distribution.
    """
    return np.exp(-(x/s)**(-alpha))

def frechet(x, s, alpha):
    """
    Probability mass function of the Frechet distribution.
    """
    return np.exp(-(x/s)**(-alpha)) * (alpha * (s ** alpha) * (x ** (alpha-1)))

def kstest(params, x, ecdf, tcdf, M, print_results = False):
    """
    Kolmogorov-Smirnov (KS) test: a goodness of fit test between an empirical CDF E(x) and a theoretical CDF T(x) of the variable x.
    The KS statistic D: Supremum of |T(x) - E(x)| across all x:
        D = sup_x |T(x) - E(x)|,
    where the subscript n indicates the number of observations/samples.
    A perfect fit between E(x) and T(x) would have a KS statistic D = 0.
    The KS statistic D follows a Kolmogoriv distribution PDF in the limit as the sample size -> infty.
    We can find the value of D that corresponds to a confidence level of alpha:
        Pr(D <= D_alpha) = 1 - alpha 
    """
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
    print(params)
    return x, frechet(x, *params), params
    
    

#%%
# dffit = crossfit(dataframe)
# diag, peri = dffit.keys()

# for i in [diag, peri]:
#     x = dffit[i]['p']
#     y = dffit[i]['d']
#     yerr = dffit[i]['d_err'] 
#     u, v, params, cov = Dfit(x, y, p0 = [1,1])
#     plt.plot(u, v)