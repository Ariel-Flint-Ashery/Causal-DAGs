from DAG_Library.module_random_geometric_graphs import _poisson_cube_sprinkling, lp_random_geometric_graph
from DAG_Library.module_path_algorithms import BFS_percolating
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

def R_BinarySearchPercolation(X, p, epsilon = 0.01):
    """
    Binary Search for a critical Radius for a given graph.
    Input:
        X: LIST of numpy.ndarray, contains a list of 1xd arrays.
            The dth element of each array represents the dth dimensional coordinate of that point.
        p: Minkowski index (the Lp space we are interested in)
        epsilon: Convergence criterion. default = 0.001
    
    Output:
        R: Critical percolation radius.
    """
    R = 1
    RL, RH = 0, 1
    while abs(RL - RH) > epsilon:
        G = lp_random_geometric_graph(X, R, p) #create new graph in each iteration, for the same set of points
        if BFS_percolating(G):
            RH = R.copy()
            R = RH - (RH - RL)/2
        else: #if BFS(R) == False
            RL = R.copy()
            R = RL + (RH - RL)/2
    
    return R #returns critical R value

def RadiusPercolation(p_val, N, density, vol, d, epsilon):
    """
    Find the critical radius for a set of p values using numerical methods.
    Currently implemented for binary search.

    Input:
        p_vals: list of Minkowski distance p indices
        N: number of graphs generated in each iteration for each p value
        density: density of points generated according to PPP
        d: dimensions of the cube
        vol: size of the d-dimensional cube
        epsilon: convergence criteria for binary search

    Output:
        crit_dict: dictionary item, {p: list of critical radii}
    
    """
    crit_dict = {}
    for p in p_val:
        R_list = [] #list of critical R values for a graph for fixed p. 
        for i in range(N): #generate N graphs
            X = _poisson_cube_sprinkling(density, vol, d)
            Rcrit = R_BinarySearchPercolation(X, p, epsilon)
            R_list.append(Rcrit)
    
        crit_dict['%s' % (p)] = R_list
    
    return crit_dict

def AnalyticCritRadius(p, d, density, deg = 1):
    """
    Calculate the anayltic solution for the critical radius for 
    connectedness for an RGG using a radius defined using the p-Minkowski
     distance, for a given degree. 
    Input:
        deg: Expected degree in connected graph. Default = 1
        p: Minkowski index (the Lp space we are interested in)
        d: dimensions of the Lp space
        density: density of points in the volume, distributed according to PPP

    Output:
        R: Critical Radius derived from the analytic solution for the d-dimensional
        p-Minkowski volume.
    """

    R = (deg/gamma(1+1/p))*(gamma(1+(d/p))/density)**(1/d)
    return R

def percolate_plot(crit_dict, bins = 100):
    """
    plot a histogram of the critical radii for each p value.

    Input:
        crit_dict: dictionary item, {p: list of critical radii}

    Output:
        overlaid matplotlib histogram plot
    """
    for item in crit_dict:
        x = crit_dict[item]
        plt.hist(x, bins, histtype=u'step', label = "p = %s" % (item), density= True)
    
    plt.xlabel('Critical Radius')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Numerical Percolation Transition for connectedness in Lp Cube Space RGGs')
    plt.savefig('Critical Radius Percolation.png')
    plt.show()  

def AnalyticRTest(p_vals, N, d, density, vol, deg =1):
    """
    Calculate probability of connectedness for analytic solution for critical
    radius in a DAG. 

    Input:
        p_vals: list of Minkowski distance p indices
        N: number of graphs generated in each iteration for each p value
        density: density of points generated according to PPP
        d: dimensions of the cube
        vol: size of the d-dimensional cube
        deg: expected outgoing degree at each node

    Output:
        percolation_probability: list of probabilities of connected graph at 
        each p value
    """

    #note: clean up code by setting d & density to **kargs or similar
    percolation_probability = []
    for p in p_vals:
        count = 0
        for i in range(N):
            X = _poisson_cube_sprinkling(density, vol, d)
            R = AnalyticCritRadius(p, d, density, deg)
            G = lp_random_geometric_graph(X, R, p)
            if BFS_percolating(G):
                count += 1
        
        if count == 0:
            percolation_probability.append(0)

        else:
            percolation_probability.append(count/N)

    return percolation_probability

def NumericalCritRadius(crit_dict):
    """
    Calculate the mean critical radius values for a set of p values.

    Input:
        crit_dict: dictionary item, {p: list of critical radii}

    Output:
        (Rcrit_val, Rcrit_val_std)
        Rcrit_val: list of mean critical radius values
        Rcrit_val_std: list of the standard deviation of critical radius values
    
    """
    Rcrit_val = [] 
    Rcrit_val_std = []
    for p in crit_dict:
        radius_mean = np.mean(np.array(crit_dict[p]))
        radius_std = np.std(np.array(crit_dict[p]))
        Rcrit_val.append(radius_mean)
        Rcrit_val_std.append(radius_std)
    
    return Rcrit_val, Rcrit_val_std

