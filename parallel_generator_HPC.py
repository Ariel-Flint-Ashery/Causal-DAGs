import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from tqdm import tqdm
import copy 
import multiprocessing

 #HPC_opt_data_rho_M
#%%
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
    file_name = os.path.join(directory, 'DAG_data_files/path_data', f'{_file_name}.pkl')
    return file_name

#%% Independent Variable
RHO = 4000
V = 1
D = 2
K = 3
M = 10000
#%% Measurement variables
dep_var = ['d', 'l','j3']
optimizer = 'geo' #or 'net'
if optimizer == 'geo':
    path_type = ['sp', 'lp', 'GreedyGeoShort', 'GreedyGeoLong', 'GreedyNetShort', 'GreedyNetLong', 'rwlk'] #['spg', 'lpg', 'gp'] or #['spn', 'lpn', 'gp']  #use __n for network optimization, __g for geometric optimization
if optimizer == 'net':
    path_type = ['sp', 'lp', 'GreedyNetShort', 'GreedyNetLong', 'rwlk']
a = np.sqrt(2)
b = 1.025
P = list(np.round([a**n for n in range(-4,5)], decimals = 5)) + list(np.round([b**n for n in range(-4,5)], decimals = 5))
P = list(set(P))
P.sort()
sprinkling_type = 'consistent' #'random' or 'consistent'
fname = 'para_%s_%s_%s_%s' % (optimizer, RHO, M, sprinkling_type)
#%% define generation functions
def generateDataframe(M = None):
    dataframe = {dv:{pt:{p:{'raw':[]} for p in P} for pt in path_type} for dv in dep_var}
    for p in P:
        for path in path_type:
            dataframe['d'][path][p]['raw'] = []
            dataframe['l'][path][p]['raw'] = []
            dataframe['j3'][path][p]['raw'] = []
            dataframe['j3'][path][p]['sum'] = []
            dataframe['j3'][path][p]['mean'] = []
            dataframe['j3'][path][p]['err'] = []

    if M != None:
        dataframe['config'] = {'constants': [RHO, V, D, K, M], 'dep_var': dep_var, 'path_types': path_type, 'optimizer': optimizer, 'sprinkling type': sprinkling_type}
    
    return dataframe

def geo_generator():
    dataframe = generateDataframe()

    _P = {p:{} for p in P}
    G = {p:{'graph_dict':{}, 'edge_list':{}} for p in P}
    
    while _P:
        if sprinkling_type == 'consistent':
            pos = rgg._poisson_cube_sprinkling(RHO, V, D, fixed_N = True)
            _P = {p:{} for p in P}
            G = {p:{'graph_dict':{}, 'edge_list':{}} for p in P}
            for p in P:
                r = pa.convert_degree_to_radius(K, RHO, D, p)
                edge_list, graph_dict = rgg.lp_random_geometric_graph(pos, r, p, show_dist = True)
                percolating = pa.DFS_percolating(graph_dict)
                if percolating == True:
                    G[p]['graph_dict'] = graph_dict
                    G[p]['edge_list'] = edge_list
                    _P.pop(p)
        
        if sprinkling_type == 'random':
            _PK = copy.deepcopy(list(_P.keys()))
            for p in _PK:
                pos = rgg._poisson_cube_sprinkling(RHO, V, D, fixed_N = True)
                r = pa.convert_degree_to_radius(K, RHO, D, p)
                edge_list, graph_dict = rgg.lp_random_geometric_graph(pos, r, p, show_dist = True)
                percolating = pa.DFS_percolating(graph_dict)
                if percolating == True:
                    G[p]['graph_dict'] = graph_dict
                    G[p]['edge_list'] = edge_list
                    _P.pop(p)

    for p in P:
        edge_list = G[p]['edge_list']
        graph_dict = G[p]['graph_dict']

        sp, lp = pa.getPaths(graph_dict, optimizer)
        rwlk = pa.random_walk(graph_dict)
        GreedyNetShort = pa.greedy_path_net(graph_dict, type = 'short')
        GreedyNetLong = pa.greedy_path_net(graph_dict, type = 'long')
        if optimizer == 'geo':
            GreedyGeoShort = pa.greedy_path_geo(graph_dict, type = 'short')
            GreedyGeoLong = pa.greedy_path_geo(graph_dict, type = 'long')
            paths = [sp, lp, GreedyGeoShort, GreedyGeoLong, GreedyNetShort, GreedyNetLong, rwlk]
        if optimizer == 'net':
            paths = [sp, lp, GreedyNetShort, GreedyNetLong, rwlk]

        paths = {path_type[i]: paths[i] for i in range(len(paths))}
        #print(paths)
        for path in path_type:
            _d, _l = pa.pathDist(graph_dict, paths[path], p)
            _J3 = pa.pathJaggy3(pos, paths[path])
            
            dataframe['d'][path][p]['raw'] = _d
            dataframe['l'][path][p]['raw'] = _l

            #dataframes take angular all angular values in the form (angle list, sum, mean, std)
            dataframe['j3'][path][p]['raw'] = _J3[0]
            dataframe['j3'][path][p]['sum'] = _J3[1]
            dataframe['j3'][path][p]['mean'] = _J3[2]
            dataframe['j3'][path][p]['err'] = _J3[3]

    return dataframe
#%% parallelise
start = time.perf_counter()
if __name__ == "__main__":
    print("""
          -----------------------------
          
              STARTING MULTIPROCESS
          
          -----------------------------
          """)
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1) #uses all available processors
    dfs = pool.starmap(geo_generator, [() for _ in range(M)]) 
    pool.close()
    pool.join()

    
#%% combine dataframes
df = generateDataframe(M)
for p in P:
    for path in path_type:
        #df[variable][path][p] = [d[variable][path][p] for d in dfs]
        df['d'][path][p]['raw'] = [d['d'][path][p]['raw'] for d in dfs]
        df['l'][path][p]['raw'] = [d['l'][path][p]['raw'] for d in dfs]
        df['j3'][path][p]['raw'] = [d['j3'][path][p]['raw'] for d in dfs]
        df['j3'][path][p]['sum'] = [d['j3'][path][p]['sum'] for d in dfs]
        df['j3'][path][p]['mean'] = [d['j3'][path][p]['mean'] for d in dfs]
        df['j3'][path][p]['err'] = [d['j3'][path][p]['err'] for d in dfs]

#%% save files
f = open(f'{file_id(fname)}', 'wb')
pickle.dump(df, f)
f.close()

print('Time elapsed: %s'% (time.perf_counter()-start))
print(sprinkling_type)

