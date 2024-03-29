# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 16:42:31 2023

@author: ariel
"""
import os
import pickle
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
    BboxConnectorPatch
from DAG_Library.custom_functions import file_id
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import DAG_Library.module_fitting_functions as ff
#%%

def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = prop_lines.copy()
        prop_patches["alpha"] = prop_patches.get("alpha", 1)*0.2


        # prop_lines = dict()
        # prop_lines['alpha'] = 0.2
        # prop_lines['ec'] = 'none' 
        # prop_lines['color'] = 'lightgrey'
        
    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                            ec = 'none', alpha = 0.2) #**prop_patches)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p



def zoom_effect(ax1, ax2, zoom_type = 'xy', **kwargs):
    """
    ax2 : the big main axes
    ax1 : the zoomed axes
    The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    if zoom_type == 'x':
        loc1a, loc2a, loc1b, loc2b = [2,3,1,4]
        
    if zoom_type == 'y':
        loc1a, loc2a, loc1b, loc2b = [3,4,2,1]
    if zoom_type == 'xy':
        loc1a, loc2a, loc1b, loc2b = [3,3,1,1]
        
    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)
    
    prop_lines = {'ec':'k', 'alpha':0.5}
    prop_lines.update(kwargs)
    
    prop_patches = kwargs.copy()
    prop_patches["ec"] = "none"
    prop_patches["alpha"] = 0.2
    prop_patches["color"] = 'lightblue'

    c1, c2, bbox_patch1, bbox_patch2, p = \
        connect_bbox(mybbox1, mybbox2,
                     loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b, #loc1a=2, loc2a=3, loc1b=1, loc2b=4
                     prop_lines=prop_lines, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p
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
#%%
#load data

#NOTE: MAKE SURE TO UNZIP HPC DATA!

fname = 'para_net_4000_10000_consistent' #odd = kevin, even = ariel
try:
    dataframe = pickle.load(open(f'{file_id(fname)}', 'rb'))
except:
    raise ValueError('NO DATAFILE FOUND')
#%%
#plotting parameters
params = {
        'axes.labelsize':20,
        'axes.titlesize':20,
        'font.size':20,
        'figure.figsize': [11,11],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)
col = iter(['green', 'blue', 'red', 'm', 'c'])
shape = iter(['^', 's', 'd', '*', '.'])
# col = iter(['green', 'blue', 'red'])#, 'm', 'c'])
# shape = iter(['^', 's', 'd'])#, '*', '.'])
zoom_type = 'x'
optimizer = dataframe['config']['optimizer'] #'G' or 'N'
path_type = list(dataframe['d'].keys())

#variables
a = np.sqrt(2)
b = 1.025
# P = list(np.round([a**n for n in range(-4,5)], decimals = 5)) + list(np.round([b**n for n in range(-4,5)], decimals = 5))
# P.sort()
P = list(dataframe['d'][path_type[0]].keys())
M = dataframe['config']['constants'][-1]
#P = list(dataframe['d']['sp'].keys())
#%%
"PLOT DISTANCE"
fig, (ax1, ax2) = plt.subplots(2,1)
#col = iter(['green', 'blue', 'red', 'm', 'c', 'k', 'brown'])
col = iter(['r', 'k'])
shape = iter(['^', 's', 'd', '*', '.', '>', 'v'])

labels = ['Shortest path', 'Longest path']


for path, label in zip(path_type[:2], labels):
    colour = next(col)
    fmt = next(shape)
    x = [p for p in P if p <= 0.91 or p >= 1.1 or p == 1]
    y = [np.average(dataframe['d'][path][p]['raw']) for p in x]
    yerr = [np.std(dataframe['d'][path][p]['raw']) for p in x]
    # ax1.plot(x, y, color = colour)
    #ax1.errorbar(x, y, yerr = yerr, label = r'$%s$' % (path), fmt = fmt, ms = 12, capsize = None, color = colour,

    ax1.errorbar(x, y, yerr = yerr, label = label, fmt = fmt, ms = 10, capsize = 10, color = colour, #label = r'$%s {%s}$' % (path, optimizer)
                 markerfacecolor = 'none', markeredgewidth = 1)

ax1.plot(np.linspace(x[0], x[-1], 100), np.array([2**(1/i) for i in np.linspace(x[0], x[-1], 100)]), label = 'diagonal')
ax1.plot(np.linspace(x[0], x[-1], 100), np.array([2 for i in range(100)]), label = 'perimeter')

ax1.set_xlabel('p')
if optimizer == 'geo':
    ax1.set_ylabel('Geometric Distance')
if optimizer == 'net':
    ax1.set_ylabel('Network Distance')
ax1.legend()
ax1.set_xscale('log', base = 2)
ax1.set_yscale('log', base = 2)

#set x ticks
x_major = mpl.ticker.LogLocator(base = 2, numticks = 10)
ax1.xaxis.set_major_locator(x_major)
x_minor = mpl.ticker.LogLocator(base = 2, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
ax1.xaxis.set_minor_locator(x_minor)
ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.tick_params(axis='x', which='minor', length = 5)
ax1.tick_params(axis='x', which='major', length = 10)
#set y ticks
y_major = mpl.ticker.LogLocator(base = 2, numticks = 10)
ax1.yaxis.set_major_locator(y_major)
y_minor = mpl.ticker.LogLocator(base = 2, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
ax1.yaxis.set_minor_locator(y_minor)
ax1.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.tick_params(axis='y', which='minor', length = 5)
ax1.tick_params(axis='y', which='major', length = 10)

#plot ax2
# our desire P range is [4:13]

col = iter(['green', 'blue', 'red', 'm', 'c'])
shape = iter(['^', 's', 'd', '*', '.'])
#for path in path_type[:-1]:

col = iter(['r', 'k'])
shape = iter(['^', 's', 'd', '*', '.', '>', 'v'])
for path, label in zip(path_type[:2], labels):
    colour = next(col)
    fmt = next(shape)
    x = [p for p in P if p >= 0.9 and p <=1.2]
    y = [np.average(dataframe['d'][path][p]['raw']) for p in x]
    yerr = [np.std(dataframe['d'][path][p]['raw']) for p in x]
    # ax2.plot(x, y, color = colour)

    #ax2.errorbar(x, y, yerr = yerr, label = r'$%s$' % (path), fmt = fmt, ms = 12, capsize = None, color = colour,
    #             markerfacecolor = 'none', markeredgewidth = 1)

# dffit = ff.swapdata(dataframe, measure = 'd')
# col = iter(['darkmagenta', 'teal'])
# linestyle = iter(['dashed', 'dotted'])
# for l in dffit:
#     colour = next(col)
#     ls = next(linestyle)
#     x = dffit[l]['p']
#     y = dffit[l]['d']
#     yerr = dffit[l]['d_err']
#     u, v, params, cov = ff.Dfit(x, y, sigma = yerr, absolute_sigma = True)
#     uu = np.array([uu for uu in u if uu < P[-5] and uu > P[4]])
#     vv = ff.Dfunc(uu, *params)
#     ax1.plot(u, v, color = colour, label = r'$fit:$ $2^{(1 - b + bp^{-a})}$', linestyle = ls, linewidth = 3)
#     ax2.plot(uu, vv, color = colour, label = r'$fit:$ $2^{(1 - b + bp^{-a})}$', linestyle = ls, linewidth = 3)

    ax2.errorbar(x, y, yerr = yerr, label = label, fmt = fmt, ms = 10, capsize = 10, color = colour,
                 markerfacecolor = 'none', markeredgewidth = 1)

dffit = ff.swapdata(dataframe, measure = 'd')
col = iter(['r', 'k'])
linestyle = iter(['dashed', 'dotted'])
for l in dffit:
    colour = next(col)
    ls = next(linestyle)
    x = dffit[l]['p']
    y = dffit[l]['d']
    yerr = dffit[l]['d_err']
    u, v, params, cov = ff.Dfit(x, y, sigma = yerr, absolute_sigma = True)
    uu = np.array([uu for uu in u if uu < P[-5] and uu > P[4]])
    vv = ff.Dfunc(uu, *params)
    # ax1.plot(u, v, color = colour, label = r'$fit:$ $2^{(1 - b + bp^{-a})}$', linestyle = ls, linewidth = 2)
    # ax2.plot(uu, vv, color = colour, label = r'$fit:$ $2^{(1 - b + bp^{-a})}$', linestyle = ls, linewidth = 2)


ax2.set_xlabel('p')
if optimizer == 'geo':
    ax2.set_ylabel('Geometric Distance')
if optimizer == 'net':
    ax2.set_ylabel('Network Distance')
ax2.legend(ncol = 2)
# ax2.set_xscale('log', base = 2)
# ax2.set_yscale('log', base = 2)
ax2.tick_params(axis='x', which='minor', length = 5)
ax2.tick_params(axis='x', which='major', length = 10)
ax2.yaxis.get_ticklocs(minor = True)
ax2.minorticks_on()
ax2.tick_params(axis='y', which='minor', length = 5)
ax2.tick_params(axis='y', which='major', length = 10)
#plot zoom in effect
zoom_effect(ax2, ax1, zoom_type)
plt.tight_layout()
plt.savefig('clean_figs/path_geo_dist.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

#%% 
" PLOT DISTANCE FRACTIONAL ERRORS FOR LONGEST AND SHORTEST PATH"

dffit = ff.swapdata(dataframe, measure = 'd')
col = iter(['darkmagenta', 'teal'])
linestyle = iter(['dashed', 'dotted'])
L = list(dffit.keys())
for i in range(len(L)-1):
    fig, (ax1, ax2) = plt.subplots(2,1)
    col = iter(['r', 'k'])
    ls = next(linestyle)
    x = dffit[L[i]]['p']
    y = dffit[L[i]]['d']
    yerr = dffit[L[i]]['d_err']
    u, v, params, cov = ff.Dfit(x, y, sigma = yerr, absolute_sigma = True)
    for path, label in zip(path_type[:2], labels):
        colour = next(col)
        if path == 'sp':
        # x = [p for p in P if p <= 0.91 or p >= 1.1 or p == 1]
        # y = [np.average(dataframe['d'][path][p]['raw'])/ff.Dfunc(p, *params) for p in x]
            x = [p for p in P if p <= 0.91 or p == 1]
        
        if path == 'lp':
            x = [p for p in P if p >= 1.1 or p == 1]
                
        y = [np.average(dataframe['d'][path][p]['raw'])/ff.Dfunc(p, *params) for p in x]
        yerr = [np.std(dataframe['d'][path][p]['raw'])/ff.Dfunc(p, *params) for p in x]
        ax1.errorbar(x, y, yerr = yerr, label = label, fmt = 'x', ms = 5, capsize = 10, color = colour,
                     markerfacecolor = 'none', markeredgewidth = 1)
        #print(path, yerr)
        x = [p for p in P if p >= 0.9 and p <= 1.11]
        y = [np.average(dataframe['d'][path][p]['raw'])/ff.Dfunc(p, *params) for p in x]
        yerr = [np.std(dataframe['d'][path][p]['raw'])/ff.Dfunc(p, *params) for p in x]
        ax2.errorbar(x, y, yerr = yerr, label = label, fmt = 'x', ms = 5, capsize = 10, color = colour,
                     markerfacecolor = 'none', markeredgewidth = 1)
    
    
    ax1.legend()
    ax1.axhline(1, linestyle = 'dotted', color = 'k')
    ax1.set_xscale('log', base = 2)
    ax1.set_ylim(0.94, 1.06)
    if optimizer == 'geo':
        ax1.set_ylabel(r'$D_{geo}$' + '/' + r'$Fit_{%s}$' % (L[i]))
        ax2.set_ylabel(r'$D_{geo}$' + '/' + r'$Fit_{%s}$' % (L[i]))
    if optimizer == 'net':
        ax1.set_ylabel(r'$D_{net}$' + '/' + r'$Fit_{%s}$' % (L[i]))
        ax2.set_ylabel(r'$D_{net}$' + '/' + r'$Fit_{%s}$' % (L[i]))
    ax1.set_xlabel('p')
    ax2.set_xlabel('p')
    #set y ticks
    ax1.yaxis.get_ticklocs(minor = True)
    ax1.minorticks_on()
    ax1.tick_params(axis='y', which='minor', length = 5)
    ax1.tick_params(axis='y', which='major', length = 10)
    #set x ticks
    x_major = mpl.ticker.LogLocator(base = 2, numticks = 10)
    ax1.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 2, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
    ax1.xaxis.set_minor_locator(x_minor)
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax1.tick_params(axis='x', which='minor', length = 5)
    ax1.tick_params(axis='x', which='major', length = 10)

    ax2.legend()
    ax2.axhline(1, linestyle = 'dotted', color = 'k')
    ax2.set_ylim(0.996, 1.004)
    ax2.tick_params(axis='x', which='minor', length = 5)
    ax2.tick_params(axis='x', which='major', length = 10)
    ax2.yaxis.get_ticklocs(minor = True)
    ax2.minorticks_on()
    ax2.tick_params(axis='y', which='minor', length = 5)
    ax2.tick_params(axis='y', which='major', length = 10)
    zoom_effect(ax2, ax1, zoom_type)
    plt.tight_layout()
    plt.savefig('clean_figs/path_geo_dist_frac_err.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
#%%
"PLOT ANGLES"

#col = iter(['green', 'blue', 'red', 'm', 'c', 'k', 'brown'])
col = iter(['r', 'k'])
shape = iter(['^', 's', 'd', '*', '.', '>', 'v'])
fig, (ax1, ax2) = plt.subplots(2,1)

#plot ax1
for path, label in zip(path_type[:2], labels):
    colour = next(col)
    fmt = next(shape)
    x = [p for p in P if p <= 0.91 or p >= 1.1 or p == 1]
    y = [np.average(dataframe['j3'][path][p]['mean']) for p in x]
    yerr = [np.average(dataframe['j3'][path][p]['err'])/np.sqrt(M) for p in x] 
    #ax1.plot(x, y, color = colour)
    #ax1.errorbar(x, y, yerr = yerr, label = r'$%s {%s}$' % (path, optimizer), fmt = '.', ms = 20, capsize = 10, color = colour)
    ax1.errorbar(x, y, yerr = yerr, label = label, fmt = fmt, ms = 10, capsize = 10, color = colour,
                 markerfacecolor = 'none', markeredgewidth = 1)
ax1.set_xlabel('p')
ax1.set_ylabel('Average Angular Deviation')
ax1.legend()
ax1.set_xscale('log', base = 2)
#set y ticks
ax1.yaxis.get_ticklocs(minor = True)
ax1.minorticks_on()
ax1.tick_params(axis='y', which='minor', length = 5)
ax1.tick_params(axis='y', which='major', length = 10)
#set x ticks
x_major = mpl.ticker.LogLocator(base = 2, numticks = 10)
ax1.xaxis.set_major_locator(x_major)
x_minor = mpl.ticker.LogLocator(base = 2, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
ax1.xaxis.set_minor_locator(x_minor)
ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.tick_params(axis='x', which='minor', length = 5)
ax1.tick_params(axis='x', which='major', length = 10)
#plot ax2
# our desire P range is [4:13]
col = iter(['r', 'k'])
shape = iter(['^', 's', 'd', '*', '.'])
for path, label in zip(path_type[:2], labels):
    colour = next(col)
    fmt = next(shape)
    x = [p for p in P if p >= 0.9 and p <= 1.11]
    y = [np.average(dataframe['j3'][path][p]['mean']) for p in x]
    yerr = [np.average(dataframe['j3'][path][p]['err'])/np.sqrt(M) for p in x] 
    #ax2.plot(x, y, color = colour)
    #ax2.errorbar(x, y, yerr = yerr, label = r'$%s {%s}$' % (path, optimizer), fmt = '.', ms = 20, capsize = 10, color = colour)
    ax2.errorbar(x, y, yerr = yerr, label = label, fmt = fmt, ms = 10, capsize = 10, color = colour,
                 markerfacecolor = 'none', markeredgewidth = 1)
ax2.set_xlabel('p')
ax2.set_ylabel('Average Angular Deviation')
ax2.legend()
# ax2.set_xscale('log')
# ax2.set_yscale('log')
ax2.yaxis.get_ticklocs(minor = True)
ax2.minorticks_on()
ax2.tick_params(axis='y', which='minor', length = 5)
ax2.tick_params(axis='y', which='major', length = 10)

ax2.xaxis.get_ticklocs(minor = True)
ax2.tick_params(axis='x', which='minor', length = 5)
ax2.tick_params(axis='x', which='major', length = 10)
#plot zoom in effect
zoom_effect(ax2, ax1, zoom_type)
plt.tight_layout()
plt.savefig('clean_figs/path_geo_angle.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

#%%
for path, label in zip(path_type[:2], labels):
    x = [p for p in P if p <= 0.91 or p >= 1.1 or p == 1]
    y = [np.average(dataframe['l'][path][p]['raw']) for p in x]
    yerr = [np.std(dataframe['l'][path][p]['raw']) for p in x]
    plt.plot(x, y, label = label)
    plt.legend()