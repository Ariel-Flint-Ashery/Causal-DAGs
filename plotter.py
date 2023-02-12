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
import numpy as np
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
    prop_patches["color"] = 'red'

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
fname = 'path_data_prelim_03' #odd = kevin, even = ariel
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
_col = ['green', 'blue']#, 'red']
zoom_type = 'x'
optimizer = dataframe['config']['optimizer'] #'G' or 'N'
path_type = ['spg', 'lpg'] #'gp'

#variables
# a = np.sqrt(2)
# b = 1.025
# P = list(np.round([a**n for n in range(-4,5)], decimals = 5)) + list(np.round([b**n for n in range(-4,5)], decimals = 5))
# P.sort()
M = dataframe['config']['constants'][-1]
P = list(dataframe['d']['spg'].keys())
#%%
"PLOT DISTANCE"
col = iter(_col)
fig, (ax1, ax2) = plt.subplots(2,1)

#plot ax1
for path in path_type:
    colour = next(col)
    x = P
    y = [np.average(dataframe['d'][path][p]['raw']) for p in P]
    yerr = [np.std(dataframe['d'][path][p]['raw']) for p in P]
    ax1.plot(x, y, color = colour)
    ax1.errorbar(x, y, yerr = yerr, label = r'$%s_{%s}$' % (path, optimizer), fmt = '.', ms = 20, capsize = 10, color = colour)

ax1.set_xlabel('p')
if optimizer == 'G':
    ax1.set_ylabel('Geometric Distance')
if optimizer == 'N':
    ax1.set_ylabel('Network Distance')
ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')

#plot ax2
# our desire P range is [4:13]
col = iter(_col)
for path in path_type:
    colour = next(col)
    x = P[4:7]
    y = [np.average(dataframe['d'][path][p]['raw']) for p in x]
    yerr = [np.std(dataframe['d'][path][p]['raw']) for p in x]
    ax2.plot(x, y, color = colour)
    ax2.errorbar(x, y, yerr = yerr, label = r'$%s_{%s}$' % (path, optimizer), fmt = '.', ms = 20, capsize = 10, color = colour)

ax2.set_xlabel('p')
if optimizer == 'G':
    ax2.set_ylabel('Geometric Distance')
if optimizer == 'N':
    ax2.set_ylabel('Network Distance')
ax2.legend()
ax2.set_xscale('log')
ax2.set_yscale('log')

#plot zoom in effect
zoom_effect(ax2, ax1, zoom_type)
plt.tight_layout()
plt.show()
#%%
"PLOT ANGLES"

col = iter(_col)
fig, (ax1, ax2) = plt.subplots(2,1)

#plot ax1
for path in path_type:
    colour = next(col)
    x = P
    y = [np.average(dataframe['j3'][path][p]['mean']) for p in P]
    yerr = [np.average(dataframe['j3'][path][p]['err'])/np.sqrt(M) for p in P] 
    ax1.plot(x, y, color = colour)
    ax1.errorbar(x, y, yerr = yerr, label = r'$%s_{%s}$' % (path, optimizer), fmt = '.', ms = 20, capsize = 10, color = colour)

ax1.set_xlabel('p')
ax1.set_ylabel('j3 average')
ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')

#plot ax2
# our desire P range is [4:13]
col = iter(_col)
for path in path_type:
    colour = next(col)
    x = P[4:7]
    y = [np.average(dataframe['j3'][path][p]['mean']) for p in x]
    yerr = [np.average(dataframe['j3'][path][p]['err'])/np.sqrt(M) for p in x] 
    ax2.plot(x, y, color = colour)
    ax2.errorbar(x, y, yerr = yerr, label = r'$%s_{%s}$' % (path, optimizer), fmt = '.', ms = 20, capsize = 10, color = colour)

ax2.set_xlabel('p')
ax2.set_ylabel('j3 average')
ax2.legend()
ax2.set_xscale('log')
ax2.set_yscale('log')

#plot zoom in effect
zoom_effect(ax2, ax1, zoom_type)
plt.tight_layout()
plt.show()