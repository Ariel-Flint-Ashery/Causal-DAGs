"""
Testing Percolation
"""
#%%
# import module_path_algorithms
# import module_random_geometric_graphs

# import .module_random_geometric_graphs 
# import .module_path_algorithms
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
from percolation_transition import R_BinarySearchPercolation, RadiusPercolation, AnalyticCritRadius, percolate_plot, AnalyticRTest, NumericalCritRadius


import numpy as np
import matplotlib.pyplot as plt

#%%

#INITIALISE CONSTANTS
d = 2 #dimensions
vol = 1 #area
density = 50 #adaptable
N_anal = 100 #number of graphs generated for each analytic iteration
p_val = [0.5, 1, 1.5, 2, 3] #p indices to test



#%%
#Probability of connectedness using analytic radius 
analytic_prob = AnalyticRTest(p_val, N_anal, d, density, vol)

# %%
