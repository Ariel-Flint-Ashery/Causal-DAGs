# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:55:15 2022

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import networkx as nx
from operator import itemgetter

params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [11,11],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

area = 1
density = 50

no_points = np.random.poisson(density * area)

xx = np.random.uniform(0,1,((no_points, 1)))
yy = np.random.uniform(0,1,((no_points, 1)))
xy_prime = np.concatenate([xx,yy], axis = 1)
xy_unsorted = np.concatenate([np.array([[0,0],[1,1]]), xy_prime])
xy = sorted(xy_unsorted, key = itemgetter(0))


# plt.scatter(xx,yy)
#node 0 has coordinates xy[0]
#cube space rule connects directed edge from x to y if x_i < y_i for all i
#%%
all_new_edges = []

def distance_check(v):
    p = 2
    R = 0.3
    distance = (np.abs(v[0])**p + np.abs(v[1])**p)**(1/p)
    if distance < R and v[0]> 0 and v[1] > 0:
        return True
    else: 
        return False
    
# R = 0.14
for u in range(no_points+2):
    u_coord = xy[u]
    new_xy = xy - u_coord
    # edge_connection = [v for v in range(no_points+2) 
    #                    if new_xy[v,0] < R and new_xy[v,0] > 0 # turn this into a function
    #                    and new_xy[v,1] < R and new_xy[v,1] > 0]
    edge_connection = [v for v in range(no_points+2)
                       if distance_check(new_xy[v]) == True]
    new_edges = [(u,v) for v in edge_connection]
    for i in new_edges:
        all_new_edges.append(i)
        
G = nx.DiGraph()
G.add_edges_from(all_new_edges)
pos = xy
fig, ax = plt.subplots()
nx.draw_networkx_nodes(G, pos, node_size = 100, ax = ax )
nx.draw_networkx_edges(G, pos, ax = ax )
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.set_xlim(-0.01,1.01)
ax.set_ylim(-0.01,1.01)
plt.show()
#%%

paths = list(nx.all_simple_paths(G, 0, no_points+1))
path_length_network = [len(v) for v in paths]
min_path_length = min(path_length_network)
max_path_length = max(path_length_network)
shortest_path = [paths[p] for p in range(len(paths)) if path_length_network[p] == min_path_length]
shortest_path_edges = [(p[i], p[i+1]) for p in shortest_path for i in range(len(p)-1)]
longest_path = [paths[p] for p in range(len(paths)) if path_length_network[p] == max_path_length]
longest_path_edges = [[(p[i], p[i+1]) for i in range(len(p)-1)] for p in longest_path]

#%%
fig, ax = plt.subplots()
nx.draw_networkx_nodes(G, pos, node_size = 100, ax = ax )
nx.draw_networkx_edges(G, pos, ax = ax , style = '--', alpha = 0.5)
# nx.draw_networkx_edges(G, pos, edgelist = shortest_path_edges, edge_color= 'red', width = 2)
for lp in longest_path_edges:
    nx.draw_networkx_edges(G, pos, edgelist = lp, edge_color = np.random.rand(3,), width = 3)


ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.set_xlim(-0.01,1.01)
ax.set_ylim(-0.01,1.01)
plt.show()

#%%
#IMPLEMENT BFS ALGORITHM
def bfs(graph, node):
  queue = []     #Initialize a queue
  visited = [] # List to keep track of visited nodes.
  bfs_path = [] #shortest path
  visited.append(node)
  queue.append(node)

  while queue:
    s = queue.pop(0) 
    #print(s, end = " ")
    bfs_path.append(s) 

    for neighbour in graph[s]:
      if neighbour not in visited:
        visited.append(neighbour)
   
        queue.append(neighbour)
  
  return bfs_path 
#%%
#RUN
net_dict = {k:list(v) for k, v in G.adjacency()}
bfs_test = bfs(net_dict, 0)
# print(len(bfs_test))
#%%
#TOPOLOGICAL SORT USING BFS & KAHN'S ALGORITHM

#create list of 'source' nodes (i.e. nodes with no incoming edges)
#find list of all nodes that have incoming edges
#connection_list = sorted({x for v in net_dict.values() for x in v})
#ele for ele in range(max(test_list)+1) if ele not in test_list
#S = [ele for ele in list(net_dict.keys()) if ele not in connection_list] #Set of all nodes with no incoming edge
#L = [] #list that will contain sorted elements

def kahn_sort(graph):
  connection_list = sorted({x for v in net_dict.values() for x in v})
  S = [ele for ele in list(graph.keys()) if ele not in connection_list]
  G = graph.copy()
  L = []
  while S: #while S is not empty, will return True
    n = S.pop(0)
    L.append(n)

    for m in G[n]:
      G[n].remove(m)
      if m not in sorted({x for v in G.values() for x in v}):
        S.append(m)
    
    return L


#%%
def bfs_3(graph):
    """
    Breadth first search from source to target
        Source is specifically node 0
        Target is specifically the last node in the network
            Note: The network must be sorted in a way that the source node has
                  no incoming edge and the target has no outgoing edges.
    
    Input:
        Graph represented as an adjacency list in the form of a python dictionary
        e.g {0: [1, 2, 3],
             1: [2, 3],
             2: [4, 6]
             ...}
    
    Output:
        Yields a generator listing all paths from the source to the target
    """
    source = 0 
    target = {len(graph) - 1}
    
    visited = dict.fromkeys([source]) 
    queue = [(v for v in G[source])] # list containing iterable generators which are the children of the node
    
    while queue:
        children = queue[-1] # picks out last iterable generator in the queue list  
        child = next(children, None) # generates the first child
        if child is None: # if there are no more child, remove the current generator from the queue
            queue.pop()
            visited.popitem()
        else:
            if child in visited: # if the child has been visited before, do nothing
                continue
            if child in target: # if the child is the target not, yield the entire path from source to target
                yield list(visited) + [child]
            visited[child] = None # record the child as a visited node
            if target - set(visited.keys()): # if the target is not yet visited, add the children of this child to the queue
                queue.append((v for v in G[child]))
            else: # if the target is visited, remove it from the visited nodes so we don't prematurely stop searching
                visited.popitem()

#%%
paths = bfs_3(net_dict)
for i in paths:
    print(i)
    
