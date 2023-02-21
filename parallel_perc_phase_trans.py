import math
import numpy as np
import DAG_Library.module_percolation_generators as pg
import DAG_Library.module_path_algorithms as pa
import matplotlib.pyplot as plt
import scipy.optimize as op
import multiprocessing
#%% define plotting parameters

params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [15,15],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

colors = iter(['firebrick', 'olivedrab', 'navy', 'forestgreen', 'tomato', 'mediumaquamarine', 'mediumblue', 'sienna', 'cornflowerblue'])
shapes = iter(['o', '^', '*', 'H', 's', 'X', 'D', 'P', '+'])

#%% constants
D = 2
P = 2
V = 1
RHO = [64, 128, 256, 512] #, 1024]
iterations = 1000
k_range = np.array([(2+i/50) for i in range(0,8)])
#%% define functions
def power_law(x, a, delta):
    y = a*np.array(x)**delta
    return y

def perc_phase_generator():
    for k in k_range:
        shape = next(shapes)
        color = next(colors)
        Pi = []
        std_Pi = []
        for rho in RHO:
            r_range = pa.convert_degree_radius(k_range, rho, D, P)
            R, Mpi, var_Mpi = pg.num_percolating(rho, V, D, P, r_range, iterations)
            std_Mpi = np.sqrt(var_Mpi)
            std_pi = {R[i]: std_Mpi[i]/iterations for i in range(len(R))}
            pi = {R[i]: Mpi[i]/iterations for i in range(len(R))}
            r = pa.convert_degree_radius(k, rho, D, P)
            #r_2 = pa.convert_degree_radius(np.array(k), rho, D, P)
            avg_edges = pg.num_degrees(rho, V, D, P, r_range, iterations)
            #avg_avg_edges = [np.average(avg_edges[i]) for i in avg_edges.keys()]
            Pi.append(pi[r])
            std_Pi.append(std_pi[r])
        params, cov = op.curve_fit(power_law, RHO, Pi, p0 = [1, 1], sigma = std_Pi)
        exponent = params[1]
        exponent_std = np.sqrt(cov[1][1])
