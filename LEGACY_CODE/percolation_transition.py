from DAG_Library.module_random_geometric_graphs import _poisson_cube_sprinkling, lp_random_geometric_graph
from DAG_Library.module_path_algorithms import BFS_percolating, DFS_percolating
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from tqdm import tqdm

def R_BinarySearchPercolation(X, p, epsilon, end, search):
    """
    Binary Search for a critical Radius for a given graph.
    Input:
        X: LIST of numpy.ndarray, contains a list of 1xd arrays.
            The dth element of each array represents the dth dimensional coordinate of that point.
        p: Minkowski index (the Lp space we are interested in)
        epsilon: Convergence criterion. default = 0.001
        end: maximum number of iterations
        search: Type traversal algorithm used. 'DFS' or 'BFS'
    Output:
        R: Critical percolation radius.
    """
    R = 1.0
    RL, RH = 0.0, 1 #np.sqrt(2)
    n = 0
    if search == None:
        raise Exception('Traversal mode not specified')
        
    if search == 'BFS':
        #print('BFS Traversal Mode', end)
        while (abs(RL - RH) > epsilon and n<end):
            G = lp_random_geometric_graph(X, R, p) #create new graph in each iteration, for the same set of points
            if BFS_percolating(G[1]):
                RH = R
                R = RH - (RH - RL)/2
            else: #if BFS(R) == False
                RL = R
                R = RL + (RH - RL)/2
            n += 1
            
    if search == 'DFS':
        #print('DFS Traversal Mode', end)
        while (abs(RL - RH) > epsilon and n<end):
            G = lp_random_geometric_graph(X, R, p) #create new graph in each iteration, for the same set of points
            if DFS_percolating(G[1]):
                RH = R
                R = RH - (RH - RL)/2
            else: #if BFS(R) == False
                RL = R
                R = RL + (RH - RL)/2
            n += 1
    #print(n)
    return R #returns critical R value

def RadiusPercolation(p_val, N, density, vol, d, search, epsilon=0.01, end = 10):
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
        print("""
              ------------------
              %s Traversal Mode
              ------------------
              """ % (search))
        for i in tqdm(range(N)): #generate N graphs
            X = _poisson_cube_sprinkling(density, vol, d)
            Rcrit = R_BinarySearchPercolation(X, p, epsilon, end, search)
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

    R = (deg/gamma(1+(1/p)))*(gamma(1+(d/p))/density)**(1/d)
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

def AnalyticRTest(p_val, N, d, density, vol, deg =1):
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
    for p in tqdm(p_val):
        count = 0
        print(p)
        R = AnalyticCritRadius(p, d, density, deg)
        print(R)
        for i in range(N):
            X = _poisson_cube_sprinkling(density, vol, d)
            G = lp_random_geometric_graph(X, R, p)
            if BFS_percolating(G[1]) == True:
                count += 1
                #print('CONNECTION')
            # else:
            #     continue
        
        #if count == 0:
        #    percolation_probability.append(0)

        #else:
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

def bastas(input_array, iterations, p, density, vol, d):
    """
    Run the Bastas et al. (2014) method for finding critical exponent and critical degree.
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.062101
    
    """
    x, degree = input_array

    R = AnalyticCritRadius(p,d,density, degree)
    pi_list = [] #for each degree
    N_list = []
    for i in tqdm(range(iterations[0])):
        N = np.random.poisson(density * vol)
        count = 0
        for j in range(iterations[1]):
            X = _poisson_cube_sprinkling(N, vol, d, fixed_N = True)
            G = lp_random_geometric_graph(X, R, p)
            if DFS_percolating(G[1]) == True:
                count += 1
        pi = count/iterations[1]
        pi_list.append(pi)
        N_list.append(N)
    pi_arr = np.array(pi_list)
    N_arr = np.array(N_list)
    # print(pi_arr, 'PI')
    # print(N_arr, 'N')
    # print('begin H calculation')

    H = pi_arr*(N_arr**x) + 1/(pi_arr*(N_arr**x))
    #print(H, 'H')
    
    LAM = 0
    for i in range(iterations[0]):
        for j in range(iterations[0]):
            if i == j:
                continue
            LAM += (H[i]- H[j])**2

    print(LAM)
    return LAM

    # LEGACY CODE FOR MULTIPLE DEGREES
    # for degree in degrees:
    #     R = AnalyticCritRadius(p,d,density, degree)
    #     pi_list = [] #for each degree
    #     N_list = []
    #     for i in range(iterations):
    #         N = np.random.poisson(density * vol)
    #         count = 0
    #         for j in range(iterations):
    #             X = _poisson_cube_sprinkling(N, vol, d, fixed_N = True)
    #             G = lp_random_geometric_graph(X, R, p)
    #             if DFS_percolating(G[1]) == True:
    #                 count += 1
    #         pi = count/iterations
    #         pi_list.append(pi)
    #         N_list.append(N)
    #     pi_dict[degree] = np.array(pi_list)
    #     N_dict[degree] = np.array(N_list)
    
    # def H(x):
    #     H_dict = {}
    #     for degree in degrees:
    #         H_dict[degree] = [(pi_dict[degree][i]*(N_dict[degree][i]**x) + 1/(pi_dict[degree][i]*(N_dict[degree][i]**x))) for i in range(iterations)]
    #     return H_dict
    
    #def Lam(x, degree, H_dict):
        #H_dict = {}
        # for degree in degrees:
        #     H_dict[degree] = [(pi_dict[degree][i]*(N_dict[degree][i]**x) + 1/(pi_dict[degree][i]*(N_dict[degree][i]**x))) for i in range(iterations)]
        
        # lam = 0
        # for i in range(iterations):
        #     for j in range(iterations):
        #         if i == j:
        #             continue
        #         lam += (H_dict[degree][i] - H_dict[degree][j])**2
    
    # x0 = []
    # res = minimize(Lam, x0, method = 'BFGS', jac = )

                
        




