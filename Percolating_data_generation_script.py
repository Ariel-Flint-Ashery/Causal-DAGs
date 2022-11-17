# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:48:49 2022

IMPORTANT: download the DAG_data_files folder into the same directory as DAG_Library to save time on running the simulations

@author: kevin
"""

import DAG_Library.module_random_geometric_graphs as mod_rgg
import DAG_Library.module_path_algorithms as mod_paths
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
import math
import DAG_Library.module_percolation_generators as pg

# params = {
#         'axes.labelsize':28,
#         'axes.titlesize':28,
#         'font.size':28,
#         'figure.figsize': [22,22],
#         'mathtext.fontset': 'stix',
#         }
# plt.rcParams.update(params)
# %%
RHO = [500, 1000, 2000]
V = 1
D = 2
P = [2, 1, 0.5]
M = 200
colors = iter(['firebrick', 'tomato', 'sienna', 'olivedrab', 'forestgreen', 'mediumaquamarine', 'navy', 'mediumblue', 'cornflowerblue'])
for p in P:
    R_intervals = pg.create_R_intervals(RHO, V, D, p)
    for rho in RHO:
        R, Mpi, var_Mpi = pg.num_percolating(rho, V, D, p, R_intervals[rho], M)
        std_pi = np.sqrt(var_Mpi)/M
        pi = Mpi/M
        p0 = [R[0]*1.5, 50]
        R_prime, pi_prime, params, _ =  pg.criticality_function_fit(R, pi, p0 = p0, sigma = None)
        R_infl = params[0]
        R_scale = pg.R_analytic(p, D, rho)
        R_res = (R - R_infl) / (R_scale)
        # pi = [pi[i] for i in range(len(R)) if R[i] > R_infl]
        # std_pi = [std_pi[i] for i in range(len(R)) if R[i] > R_infl]
        # R_res = np.array([r for r in R_res if r > 0 and r < 0.5])
        color = next(colors)
        plt.errorbar(R_res,  pi,  color = color, label = rf'$\rho$ = {rho}, p = {p}')
        print(params)
        plt.legend()
        plt.xlabel(r"rescaled R' = $\frac{R - R_{inflection}}{R_{analytic}(\rho, p)}$")
        plt.ylabel(r"$\pi$(R')")
