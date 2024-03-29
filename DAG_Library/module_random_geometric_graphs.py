# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 01:41:47 2022

@author: kevin & Ariel 
"""

import numpy as np
from operator import itemgetter

def _poisson_cube_sprinkling(density, vol, d, fixed_N = False):
    """
    Generate a set of geometric points in a d-dimensional cube with a 'volume' of vol at a specified density
    
    Inputs:
        density: float, specifies the density of points in the d-dimensional space.
        vol: float, the generalised d-dimensional volume of the d-dimensional cube
        d: INT, the dimension of the system.
        fixed_N: bool, choose fix number of nodes in graph. If True, number of nodes
                determined through Poisson Point Process. Default = False (due to legacy code)
    
    Outputs:
        coords_array_sorted: LIST of numpy.ndarray, contains a list of 1xd arrays.
                             The dth element of each array represents the dth dimensional coordinate of that point.
    """
    no_points = density
    if fixed_N == False:
        no_points = np.random.poisson(density * vol)
    
    coords = {}
    for d in range(d):
        coords[d] = np.concatenate([np.random.uniform(0, 1, ((no_points-2, 1))), np.array([[0], [1]])])
    
    coords_array = np.concatenate([x for x in coords.values()], axis = 1)
    coords_array_sorted = sorted(coords_array, key = itemgetter(0))
    return coords_array_sorted

def minkDist(v,p):
    """
    Recover the Minkowski distance between a vertex vector and the origin.
    """
    # distance = np.power(np.sum([np.abs(v[i])**p for i in range(len(v))]), 1/p)
    distance = (sum([abs(v[i])**p for i in range(len(v))]))**(1/p)
    return distance    

def _fixed_lp_distance_connection(v, R, p):
    """
    A function that checks whether a vector v is greater or less than a distance R away from the origin,
    with the distance defined as the L-p distance.
    
    Inputs:
        v: numpy.ndarray, a 1xd array representing the coordinates of v.
        R: float, the generalised radius of the circle in L-p space.
        p: float, the parameter p in L-p distance.
        
    Outputs:
        True: if v is less than a distane R away from the origin.
        False: if v is greater than a distance R away from the origin.
    """
    # distance = np.power(np.sum([np.abs(v[i])**p for i in range(len(v))]), 1/p)
    distance = (sum([abs(v[i])**p for i in range(len(v))]))**(1/p)
    if distance < R and (v>0).all():
        return True
    else:
        return False
    
def lp_random_geometric_graph(X, R, p, show_dist = True):
    """
    Creates a random geometric graph in L-p space with a specified density of points in a 
    generalised d-dimensional cube with 'volume' vol. Edges are connected under a fixed condition; 
    if the distance between two nodes is less than R, measured in L-p distances, then they are connected.
    X: set of points generated by Poisson cube sprinkling 
    """
    
    no_points = len(X)
    edge_list = []
    adjacency_list = dict.fromkeys(range(len(X)), None)
    
    for u in range(no_points):
        U = X[u]
        X_prime = X - U
        u_max = u + 1
        for Y in X_prime[u+1:, 0]:
            u_max +=1
            if Y >R:
                break
            else:
                continue
        if show_dist == True:
            edge_trigger = {v: dist for v in range(u + 1, u_max) for dist in [minkDist(X_prime[v], p)] if dist < R and (X_prime[v]>0).all()}
        else:
            edge_trigger = {v: {} for v in range(u + 1, u_max) if _fixed_lp_distance_connection(X_prime[v], R, p) == True}
        new_edges = [(u,v) for v in edge_trigger]
        
        adjacency_list[u] = edge_trigger
        for i in new_edges:
            edge_list.append(i)
    return edge_list, adjacency_list

def reduce_graph(graph_dict, r):
    """
    Removes edges from graph_dict that have geometric length > r.
    CAUTION: Ensure this is only done for graph_dict for constant p values.
    
    Inputs:
        graph_dict: dict object representing the graph adjacency list created from lp_random_geometric_graph(show_dist = True)
        r: new threshold radius
    Outputs:
        graph_dict: dict object representing the new graph adjacency list 
    """
    remove = [(k, c) for k, v in graph_dict.items() for c, d in v.items() if d > r]
    while remove:
        _n, _c = remove.pop(0)
        del graph_dict[_n][_c]
    return graph_dict
