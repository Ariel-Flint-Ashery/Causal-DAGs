# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 00:18:56 2023

@author: kevin
"""

import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.special import factorial

### Changing the plotting parameters to get a nice visualisation
params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':20,
        'figure.figsize': [11,14],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

#%% prevent file from re-generating the data; do not run the whole file, just run one cell at a time
try:
    type(initialise) # intentionally error, initialise will be defined due to except:
    print('Already initialised')
except:
    initialise = 1
#%% Parameters 
D = [500, 1000, 2000] # density
P = np.round([np.sqrt(2)**a for a in range(-2,3)], decimals = 3) # p values
M = 100 # number of iterations 
K = 3 # k_expected aka the theoretically expected value of the average degree
#%%
if initialise == 1:
    degree_dict = {d:{p:[] for p in P} for d in D}
    for i in range(M):
        for d in D:
            pos = rgg._poisson_cube_sprinkling(d, 1, 2, fixed_N = True)
            for p in P:
                print(f"""
                =========================================================
                    GENERATING GRAPHS ITERATION {i}: D = {d} ; P = {p}
                =========================================================
                """)
                R = pa.convert_degree_to_radius(K, d, 2, p)
                edge_list, graph_dict = rgg.lp_random_geometric_graph(pos, R, p, show_dist = False)
                degree_array = [len(graph_dict[u]) for u in graph_dict]
                
                degree_dict[d][p].append(degree_array)
    initialise = 0 # prevent running the data collection again and erasing degree_dict
else:
    None
#%% obtaining average degrees and population standard deviation from degree_dict
avg_degree_dict = {d: {p:{'avg':None, 'var': None, 'std': None} for p in P} for d in D}
for d in D:
    for p in P:
        avg_degree_dict[d][p]['avg'] = [np.average(deg) for deg in degree_dict[d][p]]
        avg_degree_dict[d][p]['var'] = [np.var(deg, ddof = 1) for deg in degree_dict[d][p]]
        avg_degree_dict[d][p]['std'] = [np.std(deg, ddof = 1) for deg in degree_dict[d][p]]
        
#%% plotting
colour = iter(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
def const(x, A): # define the curve y(x) = A
    return x*0 + A

for d in D[:]:
    col = next(colour)
    x = P * (1 + np.log(d)/50)
    y = np.array([np.average(avg_degree_dict[d][p]['avg']) for p in P]) + 2 * K / np.sqrt(d) # correction term = 2K/root(d)
    yerr = [np.sqrt(np.average(avg_degree_dict[d][p]['var']))/np.sqrt(M) for p in P]
    # yerr = [(np.average(np.sqrt(avg_degree_dict[d][p]['var'])))/np.sqrt(M) for p in P]

    
    ### fitting the data to a straight line with 0 gradient
    params, cov = op.curve_fit(const, x, y, p0 = K, sigma = yerr)
    xfit = np.linspace(x[0], x[-1], 1000, endpoint = True)
    yfit = const(xfit, *params)
    yfiterr = np.sqrt(cov[0][0])
    fit_legend = r'fitted with 3 $\sigma$ bounds'
    
    ### plotting the data points + fit to 3 sigma bounds
    plt.errorbar(x,  y, yerr = yerr, fmt = 'x', capsize = 5, color = col, label = rf'$\rho = {d}$')
    plt.plot(xfit, yfit, linestyle = '--', alpha = 0.7, color = col)
    plt.fill_between(xfit, yfit + 3*yfiterr, yfit - 3*yfiterr, alpha = 0.2, color = col, 
                     label = rf'$\rho = {d}$ {fit_legend}')
    
plt.axhline(K, linestyle = 'dotted', color = 'black', label = r'$\langle k_{exp} \rangle$') # plot the expected value
plt.xscale('log') 
plt.xlabel(r'$p$')
plt.ylabel(r'$\langle k \rangle$')
plt.legend(ncol = 2)
# plt.ylim(2.3)
plt.title(r'measured $\langle k \rangle$ + $\frac{2 \langle k_{exp} \rangle}{\sqrt{\rho}}$') # labelling correction term
plt.show()

#%%
def list_sum(lists):
    L = []
    for l in lists:
        L = L + l
    return L

def poisson(x, mu):
    return np.exp(-mu) * np.power(mu, x)/factorial(x)

colour = iter(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])

all_degree_dict = {}
for d in D[2]:
    all_degree_dict[d] = list_sum(degree_dict[d][p])
    binh, bine, __ = plt.hist(all_degree_dict[d], bins = np.arange(0, 15, 1))
    xfit = np.arange(0, 14, 1)
    popt, cov = op.curve_fit(poisson, xfit, binh/np.sum(binh))
    k = np.linspace(bine[0], bine[-1], 1000)
    yfit = np.sum(binh)*poisson(k, *popt)
    plt.plot(k, yfit, label = r'poisson(k, $\mu = {popt[0]} \pm {np.sqrt(np.diag(cov))[0]}$)')
    plt.xlabel('k')
    plt.ylabel('count')
    plt.legend()
    plt.show()
    
#%%
for d in D:
    print(np.sqrt(np.sum(avg_degree_dict[d][p]['var']) * 1/M))
    print(np.average(avg_degree_dict[d][p]['std']))