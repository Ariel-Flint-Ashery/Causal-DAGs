# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:56:09 2023

@author: ariel
"""

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
from DAG_Library.custom_functions import intersection
import copy

#%%
params = {
        'axes.labelsize':16,
        'axes.titlesize':28,
        'font.size':20,
        'figure.figsize': [11,11],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)
#%%
mpl.rcParams.update(mpl.rcParamsDefault)
#%%
"PLOT UNIT CIRCLES"
def lp_circle(x, p, R=1):
    y = np.power((R**p - x**p), (1/p))
    return y

def unit_circle(rmin, rmax, N, p, R=1):
    x,y = np.linspace(0,rmax, N), np.linspace(0,rmax, N)
    y = lp_circle(x, p ,R)
#%%
#constants
N = 10000
x = np.linspace(0,1, N)
P = [0.5, 1, 2, 4, np.inf]
L = 4

#create figure
fig, ax = plt.subplots(1,1, figsize = (15,15))
#add centred spline
ax.spines[["left", "bottom"]].set_position(("data", 0))
# Hide the top and right spines.
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_linestyle('--')
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, ms = 10)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, ms = 10)
ax.plot(np.linspace(0,1.1, N), [0]*N, lw = 4, c = 'k')
ax.plot([0]*N, np.linspace(0,1.1, N), lw = 4, c = 'k')
ax.set_xlim(left = -1.1 , right = 1.1)
ax.set_ylim(bottom = -1.1, top = 1.1)
ax.tick_params(axis='both', which='major', pad=15, labelsize = 14)
#uncomment for only integer ticks
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

#set colours
#cols = ['darkorchid', 'mediumslateblue', 'royalblue', 'forestgreen', 'chocolate', 'orangered']
cols = ['darkorchid', 'royalblue', 'forestgreen', 'chocolate', 'orangered']
ls = ['dotted', 'dashed', 'solid', 'dashdot', 'densely dashdotdotted']
# normalize = mpl.colors.Normalize(vmin=min(P), vmax=max(P))
# cmap = mpl.cm.get_cmap('rainbow')

#plot
for p,col,l in zip(P, cols, ls):    
    y = lp_circle(x, p)
    if p == np.inf:
        ax.plot(x,y, c = col, dashes = [8, 4, 2, 4, 2, 4], lw = L)#cmap(normalize(p)))
        ax.plot(-1*x,y, c = col, dashes = [8, 4, 2, 4, 2, 4], lw = L)
        ax.plot(-1*x,-1*y, c = col, dashes = [8, 4, 2, 4, 2, 4], lw = L)
        ax.plot(x,-1*y, c = col, dashes = [8, 4, 2, 4, 2, 4], lw = L)
        ax.plot([-1]*N, np.linspace(-1,1, N), c = col, dashes = [8, 4, 2, 4, 2, 4], lw = L)
        ax.plot([1]*N, np.linspace(-1,1, N), c = col, dashes = [8, 4, 2, 4, 2, 4], lw = L) 
        #ax.plot(-1*x,y, c = col, dashes = [8, 4, 2, 4, 2, 4])
    else: 
        ax.plot(x,y, c = col, ls = l, lw = L)
        ax.plot(-1*x, y, c = col, ls = l, lw = L)
        ax.plot(-1*x, -1*y, c = col, ls = l, lw = L)
        ax.plot(x, -1*y, c = col, ls = l, lw = L)
    
    ax.annotate('p=%s' % (p), np.array(intersection(x,y,x,x)) + np.array([[-0.01],[0.03]]),c = col, fontsize = 28)
#plt.axis('off')   

plt.show()
#%%
# ax.plot(x, y)
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# ax.spines['top'].set_color('none')

# # remove the ticks from the top and right edges
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

#%%
"PLOT CONNECTION KERNEL"

#constants
N = 8
R = 0.8
p = 2
L = 2.5
#%%
#create graph
percolating = False
while percolating == False:
    pos = rgg._poisson_cube_sprinkling(N, 1, 2, fixed_N = True)
    edge_list, graph_dict = rgg.lp_random_geometric_graph(pos, R, p)
    percolating = pa.DFS_percolating(graph_dict)


#create interval dict
interval_dict, _ = pa.getInterval(graph_dict)
pos_temp = []
i = 0
for n in graph_dict.keys():
    if n in interval_dict.keys():
        pos_temp.append(pos[n])
        

edge_temp, graph_temp = rgg.lp_random_geometric_graph(pos_temp, R, 2)

#%%
#create figure
fig, ax = plt.subplots(1,1, figsize = (15,15))
ax.set_xlim(left = -0.025 , right = 1.03)
ax.set_ylim(bottom = -0.025, top = 1.03)

#draw connection kernel
mrkr = R - 0.02
ax.plot(mrkr, 0, ">r", transform=ax.get_yaxis_transform(), clip_on=False, ms = 14)
ax.plot(0, mrkr, "^r", transform=ax.get_xaxis_transform(), clip_on=False, ms = 14)
ax.plot(np.linspace(0,mrkr, 10000), [0]*10000, lw = 4, c = 'r', ls = '--')
ax.plot([0]*10000, np.linspace(0,mrkr, 10000), lw = 4, c = 'r', ls = '--')

x = np.linspace(0,1, 10000)
y = lp_circle(x, p, R)
ax.plot(x,y, c = 'r', ls = '--', lw = 4)

#check if node lies in connection kernel
ncolors = ['r']+['k']*(len(graph_temp)-1)
ecolors = ['k']*len(edge_temp)

for n in graph_temp[0]:
    ncolors[n] = 'r'
    ecolors[edge_temp.index((0,n))] = 'r'


#draw graph
G = nx.DiGraph(edge_temp)
nx.draw_networkx(G, pos_temp, arrows = True, ax = ax, node_color = ['r']+['none']*(len(graph_temp)-1), 
                 edgecolors = ncolors, edge_color = ecolors, width = 1,
                 arrowstyle = 'simple', arrowsize = 24, node_size = 1400, linewidths = 4,
                 font_size = 28)


ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.tick_params(axis='both', which='major', pad=20, labelsize = 14)
ax.spines[["left", "bottom"]].set_position(("data", 0))
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_alpha(0.7)
ax.annotate("Connection Kernel (p=%s)" % (p),
            (0.35, 0.08), c = 'r', fontsize = 28)
plt.show()

#%%
"PLOT PATHS"
def path_node_to_edges(path_list_in_nodes):
    path_list_in_edges = [(path_list_in_nodes[i], path_list_in_nodes[i+1]) for i in range(len(path_list_in_nodes) - 1)]
    return path_list_in_edges

def plt_edges(edge_list, pos, paths, ax, labels = None, node_size = 50, show_nodes = False, **edge_kwargs):
    G = nx.DiGraph(edge_list)
    
    colors = iter(mpl.colors.TABLEAU_COLORS)
    labels = iter(labels)
    lines = iter(['--', '-.', ':'])
    ax.plot(np.linspace(0,1, 100),np.linspace(0,1, 100), ls = '-', color = 'k', label = 'pseudo-geodesic')
    for path in paths:
        color = next(colors, 'black')
        label = next(labels)
        ls = next(lines)
        path = path_node_to_edges(path)
        nx.draw_networkx_edges(G, pos, path, arrows = False, label = label, edge_color = color, style = ls, ax = ax, **edge_kwargs)
        #ax.legend()
    if show_nodes == True:    
        nx.draw_networkx_nodes(G, pos, node_size = node_size, ax = ax, node_color = 'k', edgecolors = None)
    ax.axis('off')
    
#%%
#constants

labels = ['short', 'long', 'greedy']
P = [0.5, 2]
V = 1
D = 2
RHO = 200
K = 3
_P = {p:{} for p in P}
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
#%%
#find network paths

#create figure
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,12))

#plot
for p, ax in zip(P, (ax1, ax2)) :    
    shortest_path, longest_path = pa.short_long_paths(G[p]['graph_dict'], edge_list = G[p]['edge_list'])
    greedy_path = pa.greedy_path(G[p]['graph_dict'])
    paths = [shortest_path, longest_path, greedy_path]
    plt_edges(G[p]['edge_list'], pos, paths, labels = labels, show_nodes = True, ax = ax, width = 2)

ax2.legend(loc = 'lower right')

#%%
#find geometric paths

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,12))

#plot
for p, ax in zip(P, (ax1, ax2)) :    
    shortest_path, longest_path = pa.getPaths(G[p]['graph_dict'], 'geo')
    greedy_path = pa.greedy_path_geo(G[p]['graph_dict'])
    paths = [shortest_path, longest_path, greedy_path]
    plt_edges(G[p]['edge_list'], pos, paths, labels = labels, show_nodes = True, ax = ax, width = 2)

ax2.legend(loc = 'lower right')





