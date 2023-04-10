import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import DAG_Library.module_random_geometric_graphs as rgg
import DAG_Library.module_path_algorithms as pa
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.special import gamma
from tqdm import tqdm
import pickle
import multiprocessing
import time
import scipy.optimize as op
#%%
def radiusBinarySearch(X, p, epsilon, end = np.inf):
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
    RL, RH = 0.0, 1
    n = 0
    while (abs(RL - RH) > epsilon and n<end):
        G = rgg.lp_random_geometric_graph(X, R, p) #create new graph in each iteration, for the same set of points
        if pa.DFS_percolating(G[1]):
            RH = R
            R = RH - (RH - RL)/2
        else: #if BFS(R) == False
            RL = R
            R = RL + (RH - RL)/2
        n += 1
    #print(n)
    return R #returns critical R value

def gamma_factor(p,d):
    return (gamma(1 + 1/p)**d)/gamma(1 + d/p)

def analyticCritRadius(p, d, density, deg = 1):
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

    g = 1/gamma_factor(p,d)
    R = (deg * g / rho)**(1/d)
    return R

#%% Independent Variable
RHO = [2**9, 2**10, 2**11, 2**12]
mrkrs = ['d', '*', '^', 's', '.']
V = 1
D = 2
M = 1000 #200
a = np.sqrt(2)
P = [a**n for n in range(-2, 8)]
epsilon = 0.0001
#%% plotting params
params = {
        'axes.labelsize':28,
        'axes.titlesize':28,
        'font.size':28,
        'figure.figsize': [20,15],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)
#%% check if already initialised
if os.path.isfile('radius_scaling_df_1000_0001.pkl'):
    df = pickle.load(open(f'radius_scaling_df_1000_0001.pkl', 'rb'))
    print("""
                -----------------------------------------
                    WARNING: EXISTING DATAFRAME FOUND 
                        
                        PROCEED TO PLOTTING STAGE
                -----------------------------------------        
                        """)
    # RHO = list(df.keys())[:-1]
    # mrkrs = ['d', '*', '^', 's', '.']
    # datakeys = list(df.keys())[-1]
    # config = df[datakeys]['constants']
    # RHO = config[0]
    # V = config[1]
    # D = config[2] 
    # K = config[3]
    # M = config[4]
    # P = config[5]
    # epsilon = config[6]
    # V = 1
    # D = 2
    # M = 1000 #200
    # a = np.sqrt(2)
    # P = [a**n for n in range(-2, 5)]
    # epsilon = 0.0001
#%%
def generateDataframe(M = None):
    dataframe = {rho: {p: [] for p in P} for rho in RHO}
    
    if M != None:
        dataframe['config'] = {'constants': f'RHO: {RHO}, V: {V}, D:{D} , P: {P}, M: {M}, epsilon: {epsilon}'}

    return dataframe
def scaling_generator():
    dataframe = generateDataframe()
    for rho in RHO:
        X = rgg._poisson_cube_sprinkling(rho, V, D, fixed_N = True)
        for p in P:
            dataframe[rho][p].append(radiusBinarySearch(X, p, epsilon))

    return dataframe

#%% parallelise

start = time.perf_counter()
if __name__ == "__main__":
    print("""
          -----------------------------
          
              STARTING MULTIPROCESS
          
          -----------------------------
          """)
    pool = multiprocessing.Pool(2) #multiprocessing.cpu_count() - 1 <-- uses all available processors
    dfs = pool.starmap(scaling_generator, [() for _ in range(M)])
    pool.close()
    pool.join()

#%% combine dataframes

df = generateDataframe(M)
for rho in RHO:
    for p in P:
        df[rho][p] = [d[rho][p] for d in dfs]

#%%
f = open(f'radius_scaling_df.pkl', 'wb')
pickle.dump(df, f)
f.close()

print('Time elapsed: %s'% (time.perf_counter()-start))
#%%
"""
---------------------------
    BEGIN PLOTTING HERE
---------------------------

"""
with open('radius_scaling_df_1000_0001.pkl', 'rb') as f:
    df = pickle.load(f)
#%%
#calculate errors
#error = epsilon/np.sqrt(M)
for rho in RHO:
    df[rho]['rho_avg'] = {p: np.average(df[rho][p]) for p in P}
    df[rho]['rho_anal_scale'] = [(df[rho]['rho_avg'][p])/analyticCritRadius(p, D, rho) for p in P] #/analyticCritRadius(p, D, rho) #**D)*rho*gamma_factor(p,D)
    df[rho]['rho_err'] = [np.sqrt(np.std(df[rho]['rho_anal_scale'], ddof = 1)**2 + epsilon**2)/np.sqrt(M) for p in P] #df[rho][p], ddof = 1)**2
    df[rho]['avg_err'] = [np.sqrt(np.std(df[rho][p], ddof = 1)**2 + epsilon**2)/np.sqrt(M) for p in P]
    df[rho]['rho_scale'] = [(df[rho]['rho_avg'][p]**D)*rho*gamma_factor(p,D) for p in P]
    
#%%
#plot
# normalize = mpl.colors.Normalize(vmin=min(RHO), vmax=max(RHO))
# cmap = mpl.cm.get_cmap('rainbow')
# #plot critical radius scaled by expected analytic radius for <k> = 1
# for rho, mrkr in zip(RHO, mrkrs):
#     plt.plot(P, df[rho]['rho_anal_scale'], c = cmap(normalize(rho)))
#     plt.errorbar(P, df[rho]['rho_anal_scale'], yerr = df[rho]['rho_err'],
#                   label = r'$\rho = %s$' % (rho), fmt = mrkr, ms = 20, capsize = 10, 
#                   color = cmap(normalize(rho)))
# plt.xlabel('p')
# plt.ylabel('Critical Radius')
# plt.legend()
# plt.colorbar(mpl.cm.ScalarMappable(norm=normalize, cmap=cmap))
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.show()

# #plot critical radius scaled by full relation
# for rho, mrkr in zip(RHO, mrkrs):
#     plt.plot(P, df[rho]['rho_scale'], c = cmap(normalize(rho)))
#     plt.errorbar(P, df[rho]['rho_scale'], yerr = df[rho]['rho_err'],
#                   label = r'$\rho = %s$' % (rho), fmt = mrkr, ms = 20, capsize = 10, 
#                   color = cmap(normalize(rho)))
# plt.xlabel('p')
# plt.ylabel('Critical Radius')
# plt.legend()
# plt.colorbar(mpl.cm.ScalarMappable(norm=normalize, cmap=cmap))
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.show()

#plot average radius
# for rho, mrkr in zip(RHO, mrkrs):
#     plt.plot(P, df[rho]['rho_avg'].values(), c = cmap(normalize(rho)))
#     plt.errorbar(P, df[rho]['rho_avg'].values(), yerr = df[rho]['avg_err'],
#                  label = r'$\rho = %s$' % (rho), fmt = mrkr, ms = 20, capsize = 10, 
#                  color = cmap(normalize(rho)))

# plt.xlabel('p')
# plt.ylabel('Critical Radius')
# plt.legend()
# plt.colorbar(mpl.cm.ScalarMappable(norm=normalize, cmap=cmap))
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.show()

#%%

fig, (ax1, ax2) = plt.subplots(1, 2)#, gridspec_kw={'height_ratios': [1, 1]})

normalize = mpl.colors.Normalize(vmin=min(RHO), vmax=max(RHO))
cmap = mpl.cm.get_cmap('rainbow')
xfit = np.linspace(P[0], P[-1], 1000)
cols = iter(['purple', 'blue', 'green', 'red'])
for rho, mrkr in zip(RHO, mrkrs):
    col = next(cols)
    #ax1.plot(P, df[rho]['rho_avg'].values(), c = cmap(normalize(rho)))
    ax1.errorbar(P, df[rho]['rho_avg'].values(), yerr = df[rho]['avg_err'],
                 label = r'$\rho = %s$' % (rho), fmt = mrkr, ms = 15, capsize = 10, 
                 color = col)#cmap(normalize(rho)))
    def func(x, a, b):
        # gamma_factor = (gamma(1 + 1/x)**D)/gamma(1 + x/D)
        # g = 1/gamma_factor
        # R = (g / rho)**(1/D)
        return a**(1/D)*analyticCritRadius(x, D, rho) + b #(a**(1/D))*R + b
    popt, cov = op.curve_fit(func, P, list(df[rho]['rho_avg'].values()), sigma = df[rho]['avg_err'])
    error = np.sqrt(np.diag(cov))
    ax1.plot(xfit, func(xfit, *popt), linestyle = '--', color = col)#cmap(normalize(rho)))
    #ax1.fill_between(xfit, func(xfit, popt[0]+error[0], popt[1]+error[1]), func(xfit, popt[0]-error[0], popt[1]-error[1]), alpha = 0.2,
    #                 color = cmap(normalize(rho)))
    #ax2.scatter(rho, np.sqrt(np.diag(cov)), marker = mrkr, color = cmap(normalize(rho)), ms = 200)

    y = np.array(list(df[rho]['rho_avg'].values()))/func(np.array(P), *popt)
    yerr = np.array(df[rho]['avg_err'])/func(np.array(P), *popt)
    
    ax2.errorbar(P, y, yerr = yerr, 
                 label = r'$\rho = %s$' % (rho), fmt = mrkr, ms = 15, capsize = 10, 
                 color = col)#cmap(normalize(rho)))

ax2.axhline(1, linestyle = 'dotted', color = 'k')

#labels
ax1.set_xlabel('p')
ax1.set_ylabel('Critical Radius')
ax2.set_xlabel('p')
ax2.set_ylabel('Critical Radius / Fit')

#set scales and axis
ax1.set_xscale('log', base = 2)
#ax1.set_yscale('log', base = 2)
ax2.set_xscale('log', base = 2)
#ax2.set_yscale('log')

ax1.yaxis.get_ticklocs(minor = True)
ax1.minorticks_on()
ax1.tick_params(axis='y', which='minor', length = 10)
ax1.tick_params(axis='y', which='major', length = 15)

ax2.yaxis.get_ticklocs(minor = True)
ax2.minorticks_on()
ax2.tick_params(axis='y', which='minor', length = 10)
ax2.tick_params(axis='y', which='major', length = 15)

x_major = mpl.ticker.LogLocator(base = 2, numticks = 10)
ax1.xaxis.set_major_locator(x_major)
x_minor = mpl.ticker.LogLocator(base = 2, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
ax1.xaxis.set_minor_locator(x_minor)
ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.tick_params(axis='x', which='minor', length = 10)
ax1.tick_params(axis='x', which='major', length = 15)

x_major = mpl.ticker.LogLocator(base = 2, numticks = 10)
ax2.xaxis.set_major_locator(x_major)
x_minor = mpl.ticker.LogLocator(base = 2, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
ax2.xaxis.set_minor_locator(x_minor)
ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax2.tick_params(axis='x', which='minor', length = 10)
ax2.tick_params(axis='x', which='major', length = 15)



#set labels
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles=handles,ncol=len(labels),loc="lower center", bbox_to_anchor=(0.5,-0.07), fontsize = 28)

#annotate
ax1.annotate("", xy=(3, 0.1), xycoords = 'data', 
            xytext=(3, 0.15), textcoords = 'data', 
            arrowprops=dict(width = 10, ec='k', fc = 'k', headwidth = 25, headlength = 25), color = 'k')
ax1.annotate("DECREASING DENSITY", xy = (3.5, 0.125), xycoords = 'data',
             xytext = (3.5, 0.125), textcoords = 'data')
plt.tight_layout()
plt.show()

#%%
fig, (ax1) = plt.subplots()
xfit = np.linspace(P[0], P[-1], 1000)
cols = iter(['purple', 'blue', 'green', 'red'])
for rho, mrkr in zip(RHO, mrkrs):
    col = next(cols)
    #ax1.plot(P, df[rho]['rho_avg'].values(), c = cmap(normalize(rho)))
    ax1.errorbar(P, df[rho]['rho_avg'].values(), yerr = df[rho]['avg_err'],
                 label = r'$\rho = %s$' % (rho), fmt = mrkr, ms = 15, capsize = 10, 
                 color = col)#cmap(normalize(rho)))
    def func(x, a, b):
        # gamma_factor = (gamma(1 + 1/x)**D)/gamma(1 + x/D)
        # g = 1/gamma_factor
        # R = (g / rho)**(1/D)
        return a**(1/D)*analyticCritRadius(x, D, rho) + b #(a**(1/D))*R + b
    popt, cov = op.curve_fit(func, P, list(df[rho]['rho_avg'].values()), sigma = df[rho]['avg_err'])
    error = np.sqrt(np.diag(cov))
    ax1.plot(xfit, func(xfit, *popt), linestyle = '--', color = col)#cmap(normalize(rho)))
    #ax1.fill_between(xfit, func(xfit, popt[0]+error[0], popt[1]+error[1]), func(xfit, popt[0]-error[0], popt[1]-error[1]), alpha = 0.2,
    #                 color = cmap(normalize(rho)))

#labels
ax1.set_xlabel('p')
ax1.set_ylabel('Critical Radius')


#set scales and axis
ax1.set_xscale('log', base = 2)
ax1.yaxis.get_ticklocs(minor = True)
ax1.minorticks_on()
ax1.tick_params(axis='y', which='minor', length = 10)
ax1.tick_params(axis='y', which='major', length = 15)

x_major = mpl.ticker.LogLocator(base = 2, numticks = 10)
ax1.xaxis.set_major_locator(x_major)
x_minor = mpl.ticker.LogLocator(base = 2, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
ax1.xaxis.set_minor_locator(x_minor)
ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.tick_params(axis='x', which='minor', length = 10)
ax1.tick_params(axis='x', which='major', length = 15)

#set labels
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles=handles,ncol=len(labels),loc="lower center", bbox_to_anchor=(0.5,-0.07), fontsize = 28)

#annotate
ax1.annotate("", xy=(3, 0.1), xycoords = 'data', 
            xytext=(3, 0.15), textcoords = 'data', 
            arrowprops=dict(width = 10, ec='k', fc = 'k', headwidth = 25, headlength = 25), color = 'k')
ax1.annotate("DECREASING DENSITY", xy = (3.5, 0.125), xycoords = 'data',
             xytext = (3.5, 0.125), textcoords = 'data')
plt.tight_layout()
plt.savefig('clean_figs/radius_bs.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

#%%
fig, (ax2) = plt.subplots()
xfit = np.linspace(P[0], P[-1], 1000)
cols = iter(['purple', 'blue', 'green', 'red'])
for rho, mrkr in zip(RHO, mrkrs):
    col = next(cols)
    #ax1.plot(P, df[rho]['rho_avg'].values(), c = cmap(normalize(rho)))

    def func(x, a, b):
        # gamma_factor = (gamma(1 + 1/x)**D)/gamma(1 + x/D)
        # g = 1/gamma_factor
        # R = (g / rho)**(1/D)
        return a**(1/D)*analyticCritRadius(x, D, rho) + b #(a**(1/D))*R + b
    popt, cov = op.curve_fit(func, P, list(df[rho]['rho_avg'].values()), sigma = df[rho]['avg_err'])
    error = np.sqrt(np.diag(cov))

    y = np.array(list(df[rho]['rho_avg'].values()))/func(np.array(P), *popt)
    yerr = np.array(df[rho]['avg_err'])/func(np.array(P), *popt)
    
    ax2.errorbar(P, y, yerr = yerr, 
                 label = r'$\rho = %s$' % (rho), fmt = mrkr, ms = 15, capsize = 10, 
                 color = col)#cmap(normalize(rho)))

ax2.axhline(1, linestyle = 'dotted', color = 'k')

ax2.set_xlabel('p')
ax2.set_ylabel('Critical Radius / Fit')

ax2.set_xscale('log', base = 2)
ax2.yaxis.get_ticklocs(minor = True)
ax2.minorticks_on()
ax2.tick_params(axis='y', which='minor', length = 10)
ax2.tick_params(axis='y', which='major', length = 15)

x_major = mpl.ticker.LogLocator(base = 2, numticks = 10)
ax2.xaxis.set_major_locator(x_major)
x_minor = mpl.ticker.LogLocator(base = 2, subs = np.arange(1.0, 10) * 0.1, numticks = 10)
ax2.xaxis.set_minor_locator(x_minor)
ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax2.tick_params(axis='x', which='minor', length = 10)
ax2.tick_params(axis='x', which='major', length = 15)

#set labels
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles=handles,ncol=len(labels),loc="lower center", bbox_to_anchor=(0.5,-0.07), fontsize = 28)
         
plt.tight_layout()
plt.savefig('clean_figs/radius_bs_err.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()   