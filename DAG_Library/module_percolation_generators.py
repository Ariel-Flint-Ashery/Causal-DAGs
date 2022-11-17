# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 23:04:34 2022

@author: kevin
"""

import DAG_Library.module_random_geometric_graphs as mod_rgg
import DAG_Library.module_path_algorithms as mod_paths
import pickle
import numpy as np
from scipy import optimize as op
import math
import os


def R_analytic(p, d, density):
    """
    Analytic value of the critical R value for a given p, d and density.
    Returns:
        (1/gamma(1+1/p))*(gamma(1+d/p)/density)^(1/d)
    """
    return (1/math.gamma(1+1/p))*(math.gamma(1+d/p)/density)**(1/d)

def percolation_file_id(name, rho, v, d, p, iterations, pkl = True, directory = None):
    """
    Returns:
        the file name for a given data set with parameters rho, v, d, p, iterations.
    """
    if directory == None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        directory = os.path.dirname(dir_path)
    else:
        directory = directory
    iterations = int(iterations)
    if pkl == True:
        pkl = 'pkl'
    else:
        pkl = pkl
    __file_name = f'{name}_rho{rho}_v{v}_d{d}_p{p}_iter{iterations}'
    _file_name = str(__file_name).replace(' ', '-').replace(',', '').replace('[', '-').replace(']','-').replace('.','-')
    file_name = os.path.join(directory, 'DAG_data_files', f'{_file_name}.pkl')
    return file_name

# def percolation_file_id_rename(name, rho, v, d, p, iterations, pkl = True):
#     iterations = int(iterations)
#     if pkl == True:
#         pkl = 'pkl'
#     else:
#         pkl = pkl
#     file = pickle.load(open(f'DAG_data_files/{name}_rho{rho}_v{v}_d{d}_p{p}_iter{iterations}.{pkl}', 'rb'))
#     _new_name = f'DAG_data_files/{name}_rho{rho}_v{v}_d{d}_p{p}_iter{iterations}'
#     new_name = str(_new_name).replace(' ', '-').replace(',', '').replace('[', '-').replace(']','-').replace('.','-')
#     f = open(f'{new_name}.{pkl}', 'wb')
#     pickle.dump(file, f)
#     f.close()
#     os.remove(f'DAG_data_files/{name}_rho{rho}_v{v}_d{d}_p{p}_iter{iterations}.{pkl}')
#     return None

#%% 
def _find_R_range(density, vol, d, p, iterations = 50):
    """
    For a given density, volume, dimension and p:
        Finds the range of values R between R_min and R_max.
        R_min is the maximum value of R that does NOT yield a percolating graph
        R_max is the minimum value of R that DOES yield a percolating graph
    Returns:
        R_min, R_max: float
    """
    iterations = int(iterations)
    poisson_point_list = []
    for m in range(iterations):
        poisson_point_list.append(mod_rgg._poisson_cube_sprinkling(density, vol, d))
    
    min_found = False
    max_found = False
    
    dr = 1/(density ** 0.5)
    R_init = R_analytic(p, d, density)
    over = False
    over_new = False
    while min_found == False:
        perc_count = 0
        for X in poisson_point_list:
            _, al = mod_rgg.lp_random_geometric_graph(X, R_init, p)
            path_status = mod_paths.DFS_percolating(al)
            if path_status == True:
                perc_count += 1
                
        print(f'perc_count = {perc_count}, R = {round(R_init, 5)}, dr = {round(dr, 5)}')
        if perc_count == 0 and over != over_new and dr < 1/density:
            R_min = R_init
            min_found = True
            
        over = over_new
        if perc_count == 0:
            R_new = R_init + dr
            over_new = False
        else:
            R_new = R_init - dr
            over_new = True
        if over_new != over:
            dr = dr * 7/10
        R_init = R_new
        
    dr = 1/(density ** 0.5)
    R_init = R_analytic(p, d, density)
    over = False
    over_new = False
    while max_found == False:
        perc_count = 0
        for X in poisson_point_list:
            _, al = mod_rgg.lp_random_geometric_graph(X, R_init, p)
            path_status = mod_paths.DFS_percolating(al)
            if path_status == True:
                perc_count += 1
                
        print(f'perc_count = {perc_count}, R = {round(R_init, 5)}, dr = {round(dr, 5)}')
        if perc_count == iterations and over != over_new and dr < 1/density:
            R_max = R_init
            max_found = True
            
        over = over_new
        if perc_count == iterations:
            R_new = R_init - dr
            over_new = True
        else:
            R_new = R_init + dr
            over_new = False
        if over_new != over:
            dr = dr * 7/10
        R_init = R_new
        
    return R_min, R_max
 
def _get_R_ranges(densities, vol, d, p, iterations = 50): #don't run this directly
    """
    For a given set of densities:
        Find the R ranges for each density.
        Stores the data in DAG_data_files in percolation_file_id() format.
    Return
        R_ranges: dict, keys = densities, values = [R_min, R_max]
    """
    iterations = int(iterations)
    fname = 'r_ranges'
    R_ranges = dict.fromkeys(densities)
    for rho in R_ranges:
        R_min, R_max = _find_R_range(rho, vol, d, p, iterations)
        R_ranges[rho] = [R_min, R_max]
    # f = open(f'DAG_data_files/r_ranges_rho{densities}_v{vol}_d{d}_p{p}_iter{iterations}.pkl', 'wb')
    f = open(f'{percolation_file_id(fname, densities, vol, d, p, iterations)}', 'wb')
    pickle.dump(R_ranges, f)
    f.close()
    return R_ranges

def get_R_ranges(densities, vol, d, p, iterations = 50, overwrite = False):  #run this ###############################################
    """
    For a given set of densities:
        Open the file containing the R ranges, OR
        Find the R ranges for each density.
    Return
        R_ranges: dict, keys = densities, values = [R_min, R_max]
    """
    iterations = int(iterations)
    fname = 'r_ranges'
    try:
        R_ranges = pickle.load(open(f'{percolation_file_id(fname, densities, vol, d, p, iterations)}', 'rb'))
    except:
        R_ranges = _get_R_ranges(densities, vol, d, p, iterations)
    return R_ranges

def create_R_intervals(densities, vol, d, p, iterations = 50, no_points = 20): 
    """
    For a given set of R ranges for varying densities:
        Create intervals in R such that the behaviour around the critical point is clearly seen.
        Number of intervals created is no_points.
        Stores the data in DAG_data_files in percolation_file_id() format.
    Return:
        R_ranges_intervals: dict, keys = densities, values = [R_min, ..., R_max], len(values) = no_points
    """
    R_ranges_intervals = {}
    R_ranges = get_R_ranges(densities, vol, d, p, iterations)
    for rho in R_ranges:
        R_min, R_max = R_ranges[rho]
        R_ranges_intervals[rho] = np.linspace(0.8*R_min, R_max*1.2, no_points, endpoint = True)
    return R_ranges_intervals

#%%
def _percolating_data_generator(density, vol, d, p, R_range, iterations):
    """
    For a given set of parameters: density, vol, d, p, R_range, iterations:
        Count the number of percolating graphs over the range R_range for M iterations.
        Count the average degree of the graphs for over the range R_range for M iterations.
    Returns:
        perc_counter: dict, keys = R, values = number of percolating graphs.
        avg_degree_counter: dict, keys = R, values = average degree of graphs.
    """
    fname_perc = 'perc_counter'
    fname_avg_degree = 'avg_degree_counter'
    R = R_range
    perc_counter = dict.fromkeys(R, 0)
    avg_degree_counter = dict.fromkeys(R, [0])
    for j in range(iterations):
        X = mod_rgg._poisson_cube_sprinkling(density, vol, d)
        for r in R:
            el, al = mod_rgg.lp_random_geometric_graph(X, r, p)
            if mod_paths.DFS_percolating(al, len(al) - 1) == True:
                perc_counter[r] += 1
                avg_degree_counter[r].append(len(el)/(2*len(X)))
        print(f'p = {p}, rho = {density}: {j}:done')
    f = open(f'{percolation_file_id(fname_perc, density, vol, d, p, iterations)}', 'wb')
    pickle.dump(perc_counter, f)
    f.close()
    g = open(f'{percolation_file_id(fname_avg_degree, density, vol, d, p, iterations)}','wb')
    pickle.dump(avg_degree_counter, g)
    g.close()
    return perc_counter, avg_degree_counter

    
def num_percolating(density, vol, d, p, R_range, iterations):
    """
    For a given set of parameters: density, vol, d, p, R_Range, iterations:
        Count the number of percolating graphs over the range R_range for M iterations.
    Returns: 
        R: np.ndarray = numpy array of R_range
        N: np.ndarray = numpy array of integers counting the number of percolating graphs
        var_N: np.ndarray =  array of floats of the variance in N
    """
    fname = 'perc_counter'
    try:
        perc_counter = pickle.load(open(f'{percolation_file_id(fname, density, vol, d, p, iterations)}', 'rb'))
    except:
        perc_counter, _ = _percolating_data_generator(density, vol, d, p, R_range, iterations)
    N = np.array(list(perc_counter.values()))
    R = np.array(list(perc_counter.keys()))
    var_N = binomial_variance(N, N/iterations)
    return R, N, var_N

def binomial_variance(n, p):
    """
    Returns:
        Binomail variance of N measurements with success rate p
    """
    q = (1-p)
    return n*p*q

def criticality_function(x, a, b):
    """
    Returns:
        Empirical criticality function exp[-(a/x)^b]
    """
    return np.exp(-(a/x)**b) 

def criticality_function_fit(x, y, p0 = None, sigma = None):
    """
    Using scipy.optimize.curve_fit() to fit the num_percolating() data to criticality_function()
    Returns:
        u: np.ndarray, x-axis values
        v: np.ndarray, y-axis values where y = criticality_function(x, *params)
        params: np.ndarray, parameters [a, b] of criticality_function()
        cov: np.ndarray, covariance matrix of [[aa, ab], [ba, bb]] of criticality_function()
    """
    if p0 == None:
        p0 = [x[0]*1.3, 1/np.abs(x[-1] - x[0])]
    params, cov = op.curve_fit(criticality_function, x, y, p0, sigma, maxfev = 10000)
    u = np.linspace(x[0], x[-1], num = 1000)
    v = criticality_function(u, *params)
    return u, v, params, cov
    