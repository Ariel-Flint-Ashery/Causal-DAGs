# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 01:21:22 2023

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def Dfunc(x, a, b):
    return 2 ** (1 - b + b*x**(-a))

def Dfit(x, y, **kwargs):
    u = np.linspace(x[0], x[-1], 1000, endpoint = True)
    params, cov = op.curve_fit(Dfunc, x, y, **kwargs)
    v = Dfunc(u, *params)
    return u, v, params, cov

def crossfit(dataframe, measure = 'd'):
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

#%%
dffit = crossfit(dataframe)
diag, peri = dffit.keys()

for i in [diag, peri]:
    x = dffit[i]['p']
    y = dffit[i]['d']
    yerr = dffit[i]['d_err'] 
    u, v, params, cov = Dfit(x, y, p0 = [1,1])
    plt.plot(u, v)