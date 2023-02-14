import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import matplotlib.pyplot as plt
import numpy as np
import pickle

fname = 'path_data_prelim_04' #odd = kevin, even = ariel
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
    file_name = os.path.join(directory, 'DAG_data_files\path_data', f'{_file_name}.pkl')
    return file_name

#%% Independent Variable
RHO = 1000
V = 1
D = 2
K = 3
M = 500
#%% Measurement variables
dep_var = ['d', 'j1', 'j2', 'j3', 's1', 's2', 'l']
path_type = ['sp', 'lp', 'gp'] #['spg', 'lpg', 'gp'] or #['spn', 'lpn', 'gp']  #use __n for network optimization, __g for geometric optimization
optimizer = 'N'
a = np.sqrt(2)
b = 1.025
P = list(np.round([a**n for n in range(-4,5)], decimals = 5)) + list(np.round([b**n for n in range(-4,5)], decimals = 5))
P = list(set(P))
P.sort()
#%%
dataframe = {dv:{pt:{p:{'raw':[]} for p in P} for pt in path_type} for dv in dep_var}
for v in dep_var[1:6]:
    for path in path_type:
        for p in P:
            dataframe[v][path][p]['mean'] = []
            dataframe[v][path][p]['err'] = []

for v in dep_var[1:4]:
    for path in path_type:
        for p in P:
            dataframe[v][path][p]['sum'] = []

dataframe['config'] = {'constants': [RHO, V, D, K, M], 'dep_var': dep_var, 'path_types': path_type, 'optimizer': optimizer}
#%%
try:
    dataframe = pickle.load(open(f'{file_id(fname)}', 'rb'))
except:
    for i in range(M):
        _P = {p:{} for p in P}
        print(f'Iteration {i}: Percolating...')
        while _P:
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
        print("""
        -----------------------------
            STARTING MEASUREMENTS
        -----------------------------
        """)
        for p in P:
            # r = pa.convert_degree_to_radius(K, RHO, D, p)
            # edge_list, graph_dict = rgg.lp_random_geometric_graph(pos, r, p)
            edge_list = G[p]['edge_list']
            graph_dict = G[p]['graph_dict']

            sp, lp = pa.short_long_paths(graph_dict) #I think Kevin's algorith is faster for network paths.
            gp = pa.greedy_path(graph_dict)
            paths = [sp, lp, gp] 
            paths = {path_type[i]: paths[i] for i in range(len(paths))}

            for path in path_type:
                _d, _l = pa.pathDist(graph_dict, paths[path], p)
                _J3 = pa.pathJaggy3(pos, paths[path])
                
                dataframe['d'][path][p]['raw'].append(_d)
                dataframe['l'][path][p]['raw'].append(_l)
                
                #dataframes take angular all angular values in the form (angle list, sum, mean, std)
                dataframe['j3'][path][p]['raw'].append(_J3[0])
                dataframe['j3'][path][p]['sum'].append(_J3[1])
                dataframe['j3'][path][p]['mean'].append(_J3[2])
                dataframe['j3'][path][p]['err'].append(_J3[3])

# %% Save file
f = open(f'{file_id(fname)}', 'wb')
pickle.dump(dataframe, f)
f.close()