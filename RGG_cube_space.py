# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:55:15 2022

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import networkx as nx

params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [11,11],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

area = 1
density = 90

no_points = np.random.poisson(density * area)

xx = np.random.uniform(0,1,((no_points, 1)))
yy = np.random.uniform(0,1,((no_points, 1)))
xy = np.concatenate([xx,yy], axis = 1)

plt.scatter(xx,yy)

#node 0 has coordinates xy[0]
#cube space rule connects directed edge from x to y if x_i < y_i for all i
all_new_edges = []

R = 0.2
for u in range(no_points):
    u_coord = xy[u]
    new_xy = xy - u_coord
    edge_connection = [v for v in range(no_points) 
                       if new_xy[v,0] < R and new_xy[v,0] > 0
                       and new_xy[v,1] < R and new_xy[v,1] > 0]
    new_edges = [(u,v) for v in edge_connection]
    for i in new_edges:
        all_new_edges.append(i)
        
G = nx.DiGraph()
G.add_edges_from(all_new_edges)
pos = xy
nx.draw(G, pos, node_size = 0.1)