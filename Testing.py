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
from DAG_Library.module_random_geometric_graphs import _poisson_cube_sprinkling, lp_random_geometric_graph
from DAG_Library.percolation_transition import R_BinarySearchPercolation, RadiusPercolation, AnalyticCritRadius, percolate_plot, AnalyticRTest, NumericalCritRadius, bastas
from DAG_Library.module_path_algorithms import BFS_percolating, DFS_percolating
import DAG_Library.module_percolation_generators as pg
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import DAG_Library.custom_functions as cf
#%%

#INITIALISE CONSTANTS
d = 2 #dimensions
vol = 1 #area
density = [250,500] #adaptable
N_anal = 100000 #number of graphs generated for each analytic iteration
N_numeric = 500
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
X = _poisson_cube_sprinkling(vol, d, density)
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
numeric_Rcrit = []
for rho in density:
    numeric_Rcrit.append(RadiusPercolation(p_val, N_numeric, rho, vol, d, search = 'DFS', end = 20, epsilon = 0.005))
#%%
for RCRIT in numeric_Rcrit:
    percolate_plot(RCRIT, bins = 20)

#%%
numeric_Rcrit_mean = []
numeric_Rcrit_std = []
for RCRIT in numeric_Rcrit:
    mean, std = NumericalCritRadius(RCRIT)
    numeric_Rcrit_mean.append(mean)
    numeric_Rcrit_std.append(std)

#%%error
analyticR = [AnalyticCritRadius(p, d, density[0]) for p in p_val]
#%%
plt.plot(p_val, analyticR,label = 'Analytic Critical Radius')
plt.errorbar(p_val, numeric_Rcrit_mean[0], yerr = numeric_Rcrit_std[0], label = 'rho = 250')
plt.errorbar(p_val, numeric_Rcrit_mean[1], yerr = numeric_Rcrit_std[1], label = 'rho = 500')
#plt.plot(p_val, numeric_Rcrit_mean, '.k', label = 'Numerical Critical Radius')
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


#%%
#FIND CRITICAL EXPONENTS
x0 = [0.05, 2.4]
res = minimize(bastas, x0, method = 'BFGS',args = ([50,250], 2, 250, 1, 2), tol = 1e-4, callback = True, options = {'disp': True})


# %%
res.x

#%%
#degree = np.linspace(1.6, 2.2, 6)
degree = np.array([1.5, 1.6, 1.9]) #console 2
#x_range = np.array([0.1, 0.3, 0.5, 1, 1.5, 2])
x_range = np.array([0.05, 0.1, 0.15]) #console 2
Z = np.zeros((len(x_range), len(degree)))
#%%
for i in tqdm(range(len(x_range))):
    for j in range(len(degree)):
        val = np.array([x_range[i], degree[j]])
        Z[i][j] = bastas(val, [30, 300], 2, 500, 1, 2)
        
#%%
z = Z
z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

c = plt.pcolormesh(degree, x_range, z, cmap ='nipy_spectral')
plt.colorbar(c)
plt.xlabel('degree')
plt.ylabel('x_range')
plt.show()
# %%
x_range = np.array([-1, -0.5, 0.5, 1])
results = []
for x in x_range:
    result = bastas([x, 1.6], [20,100], 2, 250, 1, 2)
    results.append(result) 

#%%
plt.plot(x_range, results)
plt.grid()
plt.xlabel('critical exponent')
plt.ylabel('lambda')
plt.show()
# %%

a, b = 1, 3.5
phi = np.linspace(3, 10, 100)
x1 = a*phi - b*np.sin(phi) #+ a*np.cos(phi)
y1 = a + b*np.cos(phi) - 2

x2 = phi
y2 = np.sin(phi)+2

x, y = cf.intersection(x1, y1, x2, y2)

plt.plot(x1, y1, c="r")
plt.plot(x2, y2, c="g")
plt.plot(x, y, "*k")
plt.show()
# %%
N = [2500, 2000, 1500]
K_range = np.linspace(1, 3, 20)
keys, vals, errs = [], [], []
for n in tqdm(N):
    key, val, err = pg.num_percolating(n, 1, 2, 2, K_range, 200, 'k')
    keys.append(key)
    vals.append(val)
    errs.append(err)

#%%
import itertools
from sklearn.metrics.pairwise import euclidean_distances
beta = np.linspace(-2,2)
combos = list(itertools.combinations(range(len(N)), 2))

for b in beta:
    for combo in combos:
        x1, y1 = keys[combo[0]], vals[combo[0]]
        x2, y2 = keys[combo[1]], vals[combo[1]]
        x, y = cf.intersection(x1, y1, x2, y2)
        points = np.column_stack((x,y))
        distance = euclidean_distances(points, points)
        epsilon = distance[np.triu_indices(distance.shape[0], k = 1)]
        if np.sum(np.all(epsilon < 0.01)) == len(points):
            break
        



