import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.special import gamma
from tqdm import tqdm

#%%
def radiusBinarySearch(X, p, epsilon, end = np.inf):
    """
    Binary Search for a critical Radius for a given graph.
    Input:
        X: LIST of numpy.ndarray, contains a list of 1xd arrays.
            The dth element of each array represents the dth dimensional coordinate of that point.
        p: Minkowski index (the Lp space we are interested in)
        epsilon: Convergence criterion. default = 0.001
        end: maximum number of iterations
        search: Type traversal algorithm used. 'DFS' or 'BFS'
    Output:
        R: Critical percolation radius.
    """
    R = 1.0
    RL, RH = 0.0, 1
    n = 0
    while (abs(RL - RH) > epsilon and n<end):
        G = rgg.lp_random_geometric_graph(X, R, p) #create new graph in each iteration, for the same set of points
        if pa.DFS_percolating(G[1]):
            RH = R
            R = RH - (RH - RL)/2
        else: #if BFS(R) == False
            RL = R
            R = RL + (RH - RL)/2
        n += 1
    #print(n)
    return R #returns critical R value

def analyticCritRadius(p, d, density, deg = 1):
    """
    Calculate the anayltic solution for the critical radius for 
    connectedness for an RGG using a radius defined using the p-Minkowski
     distance, for a given degree. 
    Input:
        deg: Expected degree in connected graph. Default = 1
        p: Minkowski index (the Lp space we are interested in)
        d: dimensions of the Lp space
        density: density of points in the volume, distributed according to PPP

    Output:
        R: Critical Radius derived from the analytic solution for the d-dimensional
        p-Minkowski volume.
    """

    R = (deg/gamma(1+(1/p)))*(gamma(1+(d/p))/density)**(1/d)
    return R

#%% Independent Variable
RHO = [250, 500, 1000, 1500]
V = 1
D = 2
M = 20 #200
a = np.sqrt(2)
P = [a**n for n in range(-2, 5)]
epsilon = 0.001
#%% plotting params
params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [11,11],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

#%%
dataframe = {rho: {p: [] for p in P} for rho in RHO}

#%%
for i in tqdm(range(M)):
    for rho in RHO:
        X = rgg._poisson_cube_sprinkling(rho, V, D, fixed_N = True)
        for p in P:
            dataframe[rho][p].append(radiusBinarySearch(X, p, epsilon))
#%%
import pickle
f = open(f'radius_scaling_df.pkl', 'wb')
pickle.dump(dataframe, f)
f.close()

#%%
#calculate errors
#error = epsilon/np.sqrt(M)
for rho in RHO:
    dataframe[rho]['rho_avg'] = {p: np.average(dataframe[rho][p]) for p in P}
    dataframe[rho]['rho_scale'] = [dataframe[rho]['rho_avg'][p]/analyticCritRadius(p, D, rho) for p in P]
    dataframe[rho]['rho_err'] = [np.sqrt(np.std(dataframe[rho][p], ddof = 1)**2 + epsilon**2)/np.sqrt(M) for p in P]
#%%
#plot
normalize = mpl.colors.Normalize(vmin=min(RHO), vmax=max(RHO))
cmap = mpl.cm.get_cmap('viridis')
for rho in RHO:
    plt.plot(P, dataframe[rho]['rho_scale'], c = cmap(normalize(rho)))
    plt.errorbar(P, dataframe[rho]['rho_scale'], yerr = dataframe[rho]['rho_err'], label = r'$\rho = %s$' % (rho), fmt = '.', ms = 20, capsize = 10, color = rho, cmap = 'viridis', norm = normalize)

plt.xlabel('density')
plt.ylabel('radius scaling')
plt.legend()
plt.colorbar()
plt.xscale('log')
plt.yscale('log')
plt.show()









            