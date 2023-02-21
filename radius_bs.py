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
import pickle
import multiprocessing
import time
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
RHO = [1000, 1500, 2000, 3000, 4000]
mrkrs = ['d', '*', '^', 's', '.']
V = 1
D = 2
M = 500 #200
a = np.sqrt(2)
P = [a**n for n in range(-2, 5)]
epsilon = 0.0001
#%% plotting params
params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [18,12],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)
#%% check if already initialised
if os.path.isfile('radius_scaling_df.pkl'):
        df = pickle.load(open(f'radius_scaling_df.pkl', 'rb'))
        print("""
                    -----------------------------------------
                        WARNING: EXISTING DATAFRAME FOUND 
                            
                            PROCEED TO PLOTTING STAGE
                    -----------------------------------------        
                            """)
#%%
def generateDataframe(M = None):
    dataframe = {rho: {p: [] for p in P} for rho in RHO}
    
    if M != None:
        dataframe['config'] = {'constants': f'RHO: {RHO}, V: {V}, D:{D} , P: {P}, M: {M}, epsilon: {epsilon}'}

    return dataframe
def scaling_generator():
    dataframe = generateDataframe()
    for rho in RHO:
        X = rgg._poisson_cube_sprinkling(rho, V, D, fixed_N = True)
        for p in P:
            dataframe[rho][p].append(radiusBinarySearch(X, p, epsilon))

    return dataframe

#%% parallelise

start = time.perf_counter()
if __name__ == "__main__":
    print("""
          -----------------------------
          
              STARTING MULTIPROCESS
          
          -----------------------------
          """)
    pool = multiprocessing.Pool(2)
    dfs = pool.starmap(scaling_generator, [() for _ in range(M)]) #uses all available processors
    pool.close()
    pool.join()

#%% combine dataframes

df = generateDataframe(M)
for rho in RHO:
    for p in P:
        df[rho][p] = [d[rho][p] for d in dfs]

#%%
f = open(f'radius_scaling_df.pkl', 'wb')
pickle.dump(df, f)
f.close()

print('Time elapsed: %s'% (time.perf_counter()-start))
#%%
#calculate errors
#error = epsilon/np.sqrt(M)
for rho in RHO:
    df[rho]['rho_avg'] = {p: np.average(df[rho][p]) for p in P}
    df[rho]['rho_scale'] = [df[rho]['rho_avg'][p]/analyticCritRadius(p, D, rho) for p in P]
    df[rho]['rho_err'] = [np.sqrt(np.std(df[rho][p], ddof = 1)**2 + epsilon**2)/np.sqrt(M) for p in P]
#%%
#plot
normalize = mpl.colors.Normalize(vmin=min(RHO), vmax=max(RHO))
cmap = mpl.cm.get_cmap('rainbow')
for rho, mrkr in zip(RHO, mrkrs):
    plt.plot(P, df[rho]['rho_scale'], c = cmap(normalize(rho)))
    plt.errorbar(P, df[rho]['rho_scale'], yerr = df[rho]['rho_err'],
                 label = r'$\rho = %s$' % (rho), fmt = mrkr, ms = 20, capsize = 10, 
                 color = cmap(normalize(rho)))

plt.xlabel('Density')
plt.ylabel('Critical Radius')
plt.legend()
plt.colorbar(mpl.cm.ScalarMappable(norm=normalize, cmap=cmap))
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()









            