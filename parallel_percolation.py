import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import scipy.optimize as op
from tqdm import tqdm
import copy 
import multiprocessing

#%%
params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [15,15],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

#%% Naming the file to save 
fname = 'percolation_data_prelim_03'

def file_id(name, pkl = True, directory = None):
    """
    Returns:
        Returns the file name with all the relevant directories
    """
    if directory == None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # dir_path = os.getcwd()
        directory = dir_path#os.path.dirname(dir_path)
    else:
        directory = directory
    if pkl == True:
        pkl = 'pkl'
    else:
        pkl = pkl
    __file_name = f'{name}'
    _file_name = str(__file_name).replace(' ', '-').replace(',', '').replace('[', '-').replace(']','-').replace('.','-')
    file_name = os.path.join(directory, 'DAG_data_files/percolation_data', f'{_file_name}.pkl')
    return file_name

#%%
#%% Set up parameters of the simulation
D = [1,2,3]
P = 2 #only implemented for Euclidean geometry
V = 1
RHO = [2**9, 2**10, 2**11, 2**12]
M = 1000
# K = np.array([(2+i/50) for i in range(-10,10)])
K = np.array([0.1,1,2,3,4,5])
#%%
def generateDataframe(M = None):
    dataframe = {d:{k:{rho: {'r':pa.convert_degree_to_radius(k, rho, d, P), 'p':[], 'sc':[], 'gwcc':[]} for rho in RHO} for k in K} for d in D}

    if M != None:
        dataframe['config'] = {'constants': [RHO, V, D, K, M, P]}
    
    return dataframe


def perc_generator():
    dataframe = generateDataframe()
    for d in D:      
        for rho in RHO:
            percolating = False
            pos = rgg._poisson_cube_sprinkling(rho, V, d, fixed_N = True)
            for k in K:
                r = dataframe[d][k][rho]['r']
                _, graph_dict = rgg.lp_random_geometric_graph(pos, r, P, show_dist = False)
                percolating = pa.DFS_percolating(graph_dict)
                dataframe[d][k][rho]['p'] = percolating

    return dataframe
#%%
start = time.perf_counter()
if __name__ == "__main__":
    print("""
          -----------------------------
          
              STARTING MULTIPROCESS
          
          -----------------------------
          """)
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1) #uses all available processors
    dfs = pool.starmap(perc_generator, [() for _ in range(M)]) 
    pool.close()
    pool.join()

#%%
df = generateDataframe(M)
for d in D:
    for rho in RHO:
        for k in K:
            df[d][k][rho]['p'] = np.sum([frame[d][k][rho] for frame in dfs])

f = open(f'{file_id(fname)}', 'wb')
pickle.dump(df, f)
f.close()

print('Time elapsed: %s'% (time.perf_counter()-start))