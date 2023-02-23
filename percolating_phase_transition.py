# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:27:22 2022

@author: kevin
"""
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import math
import numpy as np
import DAG_Library.module_percolation_generators as pg
import matplotlib.pyplot as plt
import scipy.optimize as op

params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [15,15],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

def convert_degree_radius(degree_array, rho, d, p):
    gamma_factor = math.gamma(1 + d/p)/(math.gamma(1 + 1/p)**d)
    R = (degree_array * gamma_factor / rho)**(1/d)
    return np.round((R), decimals = 16)

#%% latest test: 1000
D = 2
P = 2
V = 1
RHO = [64, 128, 256, 512] #, 1024]
iterations = 1000
k_range = np.array([(2+i/50) for i in range(0,8)])
#%%

for rho in RHO:
    r_range = convert_degree_radius(k_range, rho, D, P)
    R, Mpi, var_Mpi = pg.num_percolating(rho, V, D, P, r_range, iterations)
    print(r_range)    
#%%

def power_law(x, a, delta):
    y = a*np.array(x)**delta
    return y

colors = iter(['firebrick', 'olivedrab', 'navy', 'forestgreen', 'tomato', 'mediumaquamarine', 'mediumblue', 'sienna', 'cornflowerblue'])
shapes = iter(['o', '^', '*', 'H', 's', 'X', 'D', 'P', '+'])

x = np.arange(RHO[0], RHO[-1]+1)

for k in k_range:
    shape = next(shapes)
    color = next(colors)
    Pi = []
    std_Pi = []
    for rho in RHO:
        r_range = convert_degree_radius(k_range, rho, D, P)
        R, Mpi, var_Mpi = pg.num_percolating(rho, V, D, P, r_range, iterations)
        std_Mpi = np.sqrt(var_Mpi)
        std_pi = {R[i]: std_Mpi[i]/iterations for i in range(len(R))}
        pi = {R[i]: Mpi[i]/iterations for i in range(len(R))}
        r = convert_degree_radius(k, rho, D, P)
        r_2 = convert_degree_radius(np.array(k), rho, D, P)
        avg_edges = pg.num_degrees(rho, V, D, P, r_range, iterations)
        avg_avg_edges = [np.average(avg_edges[i]) for i in avg_edges.keys()]
        Pi.append(pi[r])
        std_Pi.append(std_pi[r])
    # print(Pi[4])
    params, cov = op.curve_fit(power_law, RHO, Pi, p0 = [1, 1], sigma = std_Pi)
    exponent = params[1]
    exponent_std = np.sqrt(cov[1][1])
    plt.plot(x, power_law(x, *params), color = color, linestyle = '--')
    plt.errorbar(RHO, Pi, yerr = std_Pi, fmt = shape, color = color, ms = 10, capsize = 10,
                 label = rf'$\langle k_{{out}} \rangle$ = {round(k, 3)}, $\beta/\nu$ = {round(exponent, 2)} $\pm$ {round(exponent_std, 3)}')
    
        
        
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('1b')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\pi$')
plt.show()

#%% Checking the behaviour; not relevant plot
for rho in RHO:
    R, Mpi, var_Mpi = pg.num_percolating(rho, V, D, P, r_range, iterations)
    plt.plot(k_range, Mpi/iterations, label = f'{rho}')
    plt.axvline(2.075)
    plt.axvline(2.1)
plt.legend()
plt.ylabel
plt.xlabel(r'$\langle k_{out} \rangle$')
plt.show()
