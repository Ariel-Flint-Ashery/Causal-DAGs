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
#%% Naming the file to save 
fname = 'percolation_data_5000_test2'
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
    file_name = os.path.join(directory, 'DAG_data_files\percolation_data', f'{_file_name}.pkl')
    return file_name
#%% Set up parameters of the simulation
D = [2] # Only look at dimension = 2; structure allows for further investigation in higher dimension if needed
P = [0.5, 1, 2] # 3 values; one p<1, one p=1, one p>1
V = 1
RHO = [2**10] #, 2**11, 2**12] # ideally 4 or more different values of RHO to see if there is any trend
M = 5000 # ideally 1000 or more, so we have a higher "resolution" when it comes to looking at derivatives
K_micro = [np.round(k,2) for k in np.arange(1, 4, 0.02)] # searches in small region around the supposed critical point
K_macro = [np.round(k,2) for k in np.arange(0.25, 6.25, 0.25)] # wide sweep to see the entire critical behaviour
K = list(set(K_micro + K_macro))
K.sort()
#%%
def generateDataframe(M = None):
    dataframe = {d:
                 {p:
                  {k:
                   {rho: 
                    {'r':pa.convert_degree_to_radius(k, rho, d, p), 'p':0}
                        for rho in RHO}
                           for k in K}
                              for p in P}
                                  for d in D}
    if M != None:
        dataframe['config'] = {'constants': [RHO, V, D, K, M, P]}
    
    return dataframe

def perc_generator():
    dataframe = generateDataframe()
    for d in D:
        for rho in RHO:
            pos = rgg._poisson_cube_sprinkling(rho, V, d, fixed_N = True)
            for p in P:
                percolating = True
                K.sort(reverse = True)
                k_max = K[0]
                r = dataframe[d][p][k_max][rho]['r']
                _, graph_dict = rgg.lp_random_geometric_graph(pos, r, p, show_dist = True)
                percolating = pa.DFS_percolating(graph_dict)
                dataframe[d][p][k_max][rho]['p'] += percolating
                for k in K[1:]:
                    if percolating == True:
                        r = dataframe[d][p][k][rho]['r']
                        graph_dict = rgg.reduce_graph(graph_dict, r)
                        percolating = pa.DFS_percolating(graph_dict)
                    else:
                        None
                    dataframe[d][p][k][rho]['p'] += percolating
    return dataframe
#%% PARALLELISE
start = time.perf_counter()
dfs = []
cpus = int(multiprocessing.cpu_count()/2)
if __name__ == "__main__":
    print("""
          -----------------------------
          
              STARTING MULTIPROCESS
          
          -----------------------------
          """)
    # pool = multiprocessing.Pool(cpus+1) # uses half + 1 of available processors
    pool = multiprocessing.Pool(multiprocessing.cpu_count()- 2) #uses all available processors 
    dfs = pool.starmap(perc_generator, [() for _ in range(M)])
    pool.close()
    pool.join()
    
#%%
df = generateDataframe(M)
for d in D:
  for p in P:
    for rho in RHO:
        for k in K:
            df[d][p][k][rho]['p'] = np.sum([frame[d][p][k][rho]['p'] for frame in dfs])
f = open(f'{file_id(fname)}', 'wb')
pickle.dump(df, f)
f.close()
print('Time elapsed: %s'% (time.perf_counter()-start))