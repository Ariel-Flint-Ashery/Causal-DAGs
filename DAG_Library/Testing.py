"""
Testing Percolation
"""
#%%
# import module_path_algorithms
# import module_random_geometric_graphs

# import .module_random_geometric_graphs 
# import .module_path_algorithms
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
from module_random_geometric_graphs import _poisson_cube_sprinkling, lp_random_geometric_graph
from percolation_transition import R_BinarySearchPercolation, RadiusPercolation, AnalyticCritRadius, percolate_plot, AnalyticRTest, NumericalCritRadius
from module_path_algorithms import BFS_percolating, DFS_percolating

import numpy as np
import matplotlib.pyplot as plt

#%%

#INITIALISE CONSTANTS
d = 2 #dimensions
vol = 1 #area
density = 500 #adaptable
N_anal = 100000 #number of graphs generated for each analytic iteration
N_numeric = 1000
p_val = [0.5, 1, 1.5, 2,2.5, 3] #p indices to test

#%%
#plot critical radius solved analytically
p_val_testing = np.linspace(0, 200, 200)
#y = np.exp(-p_val_testing)
crit_R = [AnalyticCritRadius(p, d, density) for p in p_val_testing]
plt.plot(p_val_testing, crit_R)
#plt.plot(p_val_testing, y)
plt.show()

#looks exponential to me

#%%
#test single graph
p = 2
X = _poisson_cube_sprinkling(density, vol, d)
R = AnalyticCritRadius(p, d, density, deg = 1)
#R = 0.7
G = lp_random_geometric_graph(X, R, p)

#%%
print(BFS_percolating(G[1]))
#%%
#Probability of connectedness using analytic radius 

analytic_prob = []
deg_list = [1]
for deg in deg_list:
    prob = AnalyticRTest(p_val, N_anal, d, density, vol, deg)
    analytic_prob.append(prob)
# %%
for i in range(len(deg_list)):
    plt.plot(p_val, analytic_prob[i], label = 'degree = %s' % (deg_list[i]))
    
plt.xlabel('p-index')
plt.ylabel('Probability of Connection')
plt.title('PROBABILITY OF CONNECTION AT ANALYTIC CRITICAL RADIUS')
plt.legend()
plt.grid()
plt.show()

#%%

#find critical radius numerically
numeric_Rcrit = RadiusPercolation(p_val, N_numeric, density, vol, d, search = 'DFS', end = 20, epsilon = 0.005)
#%%
percolate_plot(numeric_Rcrit, bins = 20)

#%%
numeric_Rcrit_mean, numeric_Rcrit_std = NumericalCritRadius(numeric_Rcrit)
analyticR = [AnalyticCritRadius(p, d, density) for p in p_val]
#%%
plt.plot(p_val, analyticR,label = 'Analytic Critical Radius')
plt.errorbar(p_val, numeric_Rcrit_mean, yerr = numeric_Rcrit_std, label = 'error')
plt.plot(p_val, numeric_Rcrit_mean, '.k', label = 'Numerical Critical Radius')
plt.xlabel('p value')
plt.ylabel('critical radius')
plt.legend()
plt.grid()
plt.title('Critical Radius: Analytic vs Numerical Solutions')
plt.show()

#%%
from scipy.optimize import curve_fit

def ModifiedRadius(p, alpha):
    y = alpha*AnalyticCritRadius(p,d,density)
    return y

#%%
popt, pcov = curve_fit(ModifiedRadius, p_val, numeric_Rcrit_mean)
plt.errorbar(p_val, numeric_Rcrit_mean, yerr = numeric_Rcrit_std, label = 'numerical results')
plt.plot(p_val, [ModifiedRadius(p, *popt) for p in p_val], label = 'fit')
plt.legend()
plt.grid()
plt.show()


