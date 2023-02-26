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

background = '#ebecf0'
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
#%% LP UNIT CIRCLE
#constants
N = 1000
x = np.linspace(0,1, N)
P = [0.5, 1, 2, 4, np.inf]
L = 4

#create figure
fig, ax = plt.subplots(1,1, figsize = (18.5, 18.5))
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
ax.tick_params(axis='both', which='major', labelbottom = False, labelleft = False)#pad=15, labelsize = 14)
#uncomment for only integer ticks
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

#set colours
#cols = ['darkorchid', 'mediumslateblue', 'royalblue', 'forestgreen', 'chocolate', 'orangered']
cols = ['darkorchid', 'royalblue', 'forestgreen', 'chocolate', 'orangered']
ls = ['dotted', 'dashed', 'solid', 'dashdot', 'densely dashdotdotted']
# normalize = mpl.colors.Normalize(vmin=min(P), vmax=max(P))
# cmap = mpl.cm.get_cmap('rainbow')
ax.annotate('1', (-0.055, 1.03), fontsize = 34)
ax.annotate('-1', (-0.07, -1.07), fontsize = 34)
ax.annotate('-1', (-1.08, -.07), fontsize = 34)
ax.annotate('1', (1.01, -.07), fontsize = 34)
#plot
for p,col,l in zip(P, cols, ls):    
    y = lp_circle(x, p)
    if p == np.inf:
        ax.plot(x,y, c = col, dashes = [8, 4, 2, 4, 2, 4], lw = L) #cmap(normalize(p)))
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
        
    if p == np.inf:
        p = r'$\infty$'
        
    ax.annotate('p=%s' % (p), np.array(intersection(x,y,x,x)) + np.array([[-0.01],[0.03]]),c = col, fontsize = 38)
#plt.axis('off)
fig.set_facecolor(background)
ax.set_facecolor(background)
plt.savefig('poster_figs/lp-unit-circle.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()
#%%
"PLOT CONNECTION KERNEL"

#constants
N = 8
R = 0.8
p = 2
L = 2.5
#%% CONNECTION KERNEL
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
        
pos_temp = [np.array(i) for i in [[0,0], [0.15, 0.5], [0.6, 0.1], [0.5, 0.4], [0.55, 0.8], [0.9, 0.45], [1,1]]]
edge_temp, graph_temp = rgg.lp_random_geometric_graph(pos_temp, R, 2)

#create figure
fig, ax = plt.subplots(1,1, figsize = (21,21))
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

arrow = mpl.patches.ArrowStyle.Simple(head_length = 1.2, head_width = 1.2, tail_width = 0.2)
# draw graph
G = nx.DiGraph()
G.add_nodes_from([n for n in graph_temp])
G.add_edges_from(edge_temp)
nx.draw_networkx(G, pos_temp, arrows = True, ax = ax, node_color = ['r']+['none']*(len(graph_temp)-1), 
                 edgecolors = ncolors, edge_color = ecolors, width = 1,
                 arrowstyle = arrow, arrowsize = 20, node_size = 2500, linewidths = 4, font_size = 44)

ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False)
ax.tick_params(axis='both', which='major', pad=25, labelsize = 44)
ax.spines[["left", "bottom"]].set_position(("data", 0))
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_alpha(0.7)
ax.annotate("Connection Kernel (p=%s)" % (p),
            (0.04, 0.84), c = 'r', fontsize = 52)
ax.annotate("R", xy = (0,0), xytext = (R-0.012,-0.045), c = 'r', fontsize = 52)
ax.annotate("R", xy = (0,0), xytext = (-0.04, R - 0.013), c = 'r', fontsize = 52)
ax.set_xticks(np.arange(0,2,1))
ax.set_yticks(np.arange(0,2,1))
fig.set_facecolor(background)
ax.set_facecolor(background)
plt.savefig('poster_figs/connection-kernel.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

#%%
"PLOT PATHS"
def path_node_to_edges(path_list_in_nodes):
    path_list_in_edges = [(path_list_in_nodes[i], path_list_in_nodes[i+1]) for i in range(len(path_list_in_nodes) - 1)]
    return path_list_in_edges

def plt_edges(edge_list, pos, paths, ax, labels = None, node_size = 50, show_nodes = False, **edge_kwargs):
    G = nx.DiGraph(edge_list)
    
    colors = iter(['#00CD6C', '#AF58BA'])#iter(mpl.colors.TABLEAU_COLORS)
    labels = iter(labels)
    lines = iter(['--', '-.', ':'])
    ax.plot(np.linspace(0,1, 100),np.linspace(0,1, 100), ls = '-', alpha = 0.7,
                        color = 'k', label = 'pseudo-geodesic', linewidth = 2.5)
    for path in paths:
        color = next(colors, 'black')
        label = next(labels)
        ls = next(lines)
        path = path_node_to_edges(path)
        nx.draw_networkx_edges(G, pos, path, arrows = False, label = label, edge_color = color, style = ls, ax = ax, **edge_kwargs)
        #ax.legend()
    if show_nodes == True:    
        nx.draw_networkx_nodes(G, pos, node_size = node_size, ax = ax, node_color = 'r', edgecolors = None)
    #ax.axis('off')
    # ax.set_xticks([], fontsize = 5)
    # ax.set_yticks([], fontsize = 5)
    ax.set_frame_on(False)
    
#%%
#constants

labels = ['short', 'long', 'pseudo-geodesic']
P = [0.5, 2]
V = 1
D = 2
RHO = 200
K = 5
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
    greedy_path = pa.greedy_path(G[p]['graph_dict'], 'geo')
    paths = [shortest_path, longest_path, greedy_path]
    plt_edges(G[p]['edge_list'], pos, paths, labels = labels, show_nodes = True, ax = ax, width = 2)

ax2.legend(loc = 'lower right')

# PATHS
#find geometric paths

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,9))

#plot
for p, ax in zip(P, (ax1, ax2)) :    
    shortest_path, longest_path = pa.getPaths(G[p]['graph_dict'], 'geo')
    gnx = nx.DiGraph(G[p]['edge_list'])
    nx.draw_networkx(gnx, pos, arrows = False, ax = ax, node_color = 'g', 
                      edge_color = 'k', width = 1, alpha = 0.2, with_labels = False,
                      node_size = 50, style = '--')
    #greedy_path = pa.greedy_path_geo(G[p]['graph_dict'])
    paths = [shortest_path, longest_path]#, greedy_path]
    plt_edges(G[p]['edge_list'], pos, paths, labels = labels, show_nodes = True, ax = ax, width = 4)
    rect = mpl.patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)
    
    ax.annotate(f'p = {p}', (0.5, -0.07), fontsize = 28, ha = 'center')
    #ax.set_xlabel(f'p = {p}', fontsize = 28)
    ax.annotate('S', (-0.01,-0.03), ha = 'right', fontsize = 24)
    ax.annotate('T', (1.01,1.01), ha = 'left', fontsize = 24)
    
# ax.legend(loc = 'lower right')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles=handles,ncol=len(labels),loc="lower center", bbox_to_anchor=(0.5,-0.07), fontsize = 28, facecolor = background, edgecolor = background)
plt.tight_layout()
fig.set_facecolor(background)
ax.set_facecolor(background)
plt.savefig('poster_figs/path-fig.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()