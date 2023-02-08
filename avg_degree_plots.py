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
D = [250, 500, 1000, 2000] # density
P = np.round([np.sqrt(2)**a for a in range(-2,3)], decimals = 3) # p values
M = 1 # number of iterations 
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
avg_degree_dict = {d: {p:{'avg':None, 'std': None} for p in P} for d in D}
for d in D:
    for p in P:
        avg_degree_dict[d][p]['avg'] = [np.average(deg) for deg in degree_dict[d][p]]
        avg_degree_dict[d][p]['std'] = [np.std(deg) for deg in degree_dict[d][p]]
        
#%% plotting
colour = iter(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
def const(x, A): # define the curve y(x) = A
    return x*0 + A

for d in D[:]:
    col = next(colour)
    x = P * (1 + np.log(d)/50)
    y = np.array([np.average(avg_degree_dict[d][p]['avg']) for p in P]) + 2 * K / np.sqrt(d) # correction term = 2K/root(d)
    yerr = [np.average(avg_degree_dict[d][p]['std'])/np.sqrt(M) for p in P]
    # yerr = [np.std(avg_degree_dict[d][p]['avg']) for p in P]
    
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
    
plt.axhline(3, linestyle = 'dotted', color = 'black', label = r'$\langle k_{exp} \rangle$') # plot the expected value
plt.xscale('log') 
plt.xlabel(r'$p$')
plt.ylabel(r'$\langle k \rangle$')
plt.legend(ncol = 2)
plt.ylim(2.3)
plt.title(r'measured $\langle k \rangle$ + $\frac{2 \langle k_{exp} \rangle}{\sqrt{\rho}}$') # labelling correction term
plt.show()