# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 23:26:37 2022

@authors: Kevin & Ariel

This is a prototype module for Directed Acyclic Graph projects.
This module contains path searching algorithms.
"""
import matplotlib.pyplot as plt
import numpy as np

def BFS_all_paths(graph_dict, target):
    """
    Breadth first search from source to target
        Source is specifically node 0
        Target is specifically the last node in the network
            Note: The network must be sorted in a way that the source node has
                  no incoming edge and the target has no outgoing edges.
    
    Input:
        G: Graph represented as an adjacency list in the form of a python dictionary
        e.g {0: [1, 2, 3],
             1: [2, 3],
             2: [4, 6]
             ...}
    
    Output:
        Yields a generator listing all paths from the source to the target
    """
    target = {target}
    
    visited = dict.fromkeys([0]) 
    queue = [iter(graph_dict[0])]
    
    d_min = None
    d_max = 0
    
    shortest_paths = []
    longest_paths = []
    
    while queue:
        children = queue[-1] # picks out last iterable generator in the queue list  
        child = next(children, None) # generates the first child
        if child is None: # if there are no more child, remove the current generator from the queue
            queue.pop()
            visited.popitem()
        else:
            if child in visited: # if the child has been visited before, do nothing
                continue
            if child in target: # if the child is the target, yield the entire path from source to target
                distance = len(visited) 
                
                
                if distance >= d_max: # if distance to child is >= the previously recorded max distance
                    if distance > d_max: # if it is greater, then forget all past longest paths and record new ones
                        longest_paths = []
                        d_max = distance
                    longest_paths.append(list(visited) + [child])
                
                elif d_min is None: # initialise min distance
                    d_min = distance
                    shortest_paths.append(list(visited) + [child])
                
                elif distance <= d_min: # if distance to child is <= the previously recorded max distance
                    if distance < d_min: # if it is less, then forget all past shortest paths and record new ones
                        shortest_paths = []
                        d_min = distance
                    shortest_paths.append(list(visited) + [child])
                
                
            visited[child] = None # record the child as a visited node
            if target - set(visited.keys()): # if the target is not yet visited, add the children of this child to the queue
                queue.append(iter(graph_dict[child]))
            else: # if the target is visited, remove it from the visited nodes so we don't prematurely stop searching
                visited.popitem()
    yield shortest_paths
    yield longest_paths

def BFS_percolating(graph_dict, target = None):
    """
    Breadth first search through a network for a path from source to target.
    Existence of a path means that the system is percolating.
    
    Input:
        G: Graph represented as an adjacency list in the form of a python dictionary
        e.g {0: [1, 2, 3],
             1: [2, 3],
             2: [4, 6]
             ...}
        
     Output:
         True if a path exists between the source and target
         False if a path does not exist between the source and target
    """
     
    if target == None:
        target = len(graph_dict) - 1

    if 0 not in graph_dict:
        # Checks if the initial source even has connections to other nodes
        print('no source connections')
        return False
    
    target = {target}
    
    visited = dict.fromkeys([0])
    queue = [iter(graph_dict[0])]
    
    while queue: 
        children = queue[-1]
        child = next(children, None)
        if child is None:
            queue.pop()
        else:
            if child in visited:
                continue
            if child in target:
                return True # Returns true the moment the first path is found
            visited[child] = None
            queue.append(iter(graph_dict[child]))
    return False # Returns false only if the while queue loop terminates, which indicates no path is found

def DFS_percolating(graph_dict, target = None):
    """
    Depth first search through a network for a path from the source to the target.
    The depth first serach also prioritises higher indexed nodes, which are likely
    to be closer to the target if graph_dict is sorted.
    
     Input:
        G: Graph represented as an adjacency list in the form of a python dictionary
        e.g {0: [1, 2, 3],
             1: [2, 3],
             2: [4, 6]
             ...}
        
     Output:
         True if a path exists between the source and target
         False if a path does not exist between the source and target
    """
    if target == None:
        target = len(graph_dict) - 1
        
    if 0 not in graph_dict:
        return False
    
    visited = dict.fromkeys([0])
    queue = [c for c in graph_dict[0]]
    
    if target in queue:
        return True
    
    while queue:
        child = queue[-1]
        queue.pop()
        if child in visited:
            continue
        elif child == target:
            return True
        else:
            visited[child] = None
            children = [c for c in graph_dict[child]]
            if target in children:
                return True
            for c in children:
                queue.append(c)
    return False

def greedy_path(graph_dict, target = None):
    if target == None:
        target = len(graph_dict) - 1
        
    visited = [0]
    queue = [0]
    
    while queue:
        print(queue)
        children = graph_dict[queue[-1]]
        queue.pop()
        
        if target in children:
            visited.append(target)
            return visited
        
        grand_children = {len(graph_dict[c].keys()):c for c in children}
        most_childs = max(grand_children)
        if most_childs == 0:
            return "No greedy path found"
        else:
            greedy_child = grand_children[most_childs]
        
        visited.append(greedy_child)
        queue.append(greedy_child)
        
        if greedy_child == target:
            return visited
        else:
            continue
        
def pathDist(graph_dict, path,p):
    queue = []
    distance = 0.0
    
    for i in range(len(path)-1):
        distance += graph_dict[path[i]][path[i+1]].values() #[-1]

    return distance, len(path) #geometric distance, network distance

def dijkstra(graph_dict, target = None):
    if target == None:
        target = len(graph_dict) - 1
    
    shortest_dict = dict.fromkeys(graph_dict.keys())
    #initialise dictionary
    for key in graph_dict.keys():
        shortest_dict[key] = {'dist': np.inf, 'bestParent': None }
    shortest_dict[0] = {'dist': 0.0, 'bestParent': None }

    queue = list(graph_dict.keys()) #[0]
    visited = []
    while queue:
        filter_dict = {shortest_dict[c]['dist']: c for c in graph_dict.keys() if c not in visited}
        u = filter_dict[min(filter_dict)]
        if u == target:
            break
        
        queue.remove(u)
        children = graph_dict[u] #nearest neighbours   
        tentativeDist = {c: shortest_dict[u]['dist'] + children[c] for c in children} #sum(shortest_dict[c]['dist'].add(children[c]))
        
        #compare tentative distances
        for c in children:
            if tentativeDist[c] < shortest_dict[c]['dist']:
                shortest_dict[c]['dist'] = tentativeDist[c]
                shortest_dict[c]['bestParent'] = u
        
        visited.append(u)

    shortestPath = []
    u = target
    if shortest_dict[u]['bestParent'] != None or u == 0:
        while u:
            shortestPath.append(u)
            u = shortest_dict[u]['bestParent']

    shortestPath.reverse()
    return shortestPath, shortest_dict[target]['dist']        

def findAllSources(graph_dict):
    """
    Find all vertices with no incoming edges in a directed acyclic graph.
    Graph is sorted in one dimension, such that neighbouring indices must connect
    if they satisfy the radius condition. 
        Input:
            graph_dict: adjacency dictionary of entire graph. 
            {i: {j,k,l....}}, j, k, l are outgoing from i.
        
        Output:
            list of all source vertices in graph_dict.
    """
    sources = []
    for v in graph_dict.keys():
        #break conditions
        if v == 0:
            continue

        #find source nodes
        counter = 0
        for i in range(v):
            if v not in graph_dict[i].keys():
                counter += 1

        if counter == v:
            sources.append(v)
    
    if len(graph_dict) in sources:
        raise ValueError('Target is a source. Check graph is complete.')
    
    return sources

def kahn_sort(graph_dict, S = None):
    """
    Run a kahn sorting algorithm on a directed acylic graph.
        Input:
            graph_dict: adjacency dictionary of entire graph.
            S: list of all source nodes.

        Output:
            L: Topologically sorted order of vertices. 
    
    """
    
    #legacy
    # connection_list = sorted({x for v in graph.values() for x in v})
    # S = [ele for ele in list(graph.keys()) if ele not in connection_list]
    if S == None:
        S = findAllSources(graph_dict)

    G = graph_dict.copy()
    L = []
    while S: #while S is not empty, will return True
        n = S.pop(0)
        L.append(n)

        for m in G[n]:
            G[n].pop(m)
            if m not in findAllSources(G):
                S.append(m)
    
    return L

def getInterval(graph_dict):
    """
    Input:
        graph_dict

    Output:
        L: Topologically sorted (by Kahn's algorithm) interval between source and target.
    """
    S = [0] #we only consider one source node
    G = graph_dict.copy()
    L = []
    while S: #while S is not empty, will return True
        n = S.pop(0)
        L.append(n)

        for m in G[n]:
            G[n].pop(m)
            if getIncomingDict(graph_dict)[m] == {}:
                S.append(m)
            # if m not in findAllSources(G):
            #     S.append(m)

    return L
def getIncomingDict(graph_dict): #we might want to move this to geometric graph module?
    """
    Return an adjacency dictionary for incoming edges.
        Input:
            graph_dict: adjacency dictionary of entire graph for edge between u and v.
            Here, keys indicate u and values indicate v.

        Output:
            incoming_dict: adjacency dictionary of entire graph for edge between u and v.
            Here, keys indicate v and values indicate u.

    """
    incoming_dict = dict.fromkeys(graph_dict.keys())
    for key in graph_dict.keys():
        incoming_dict[key] = {}
    
    for v in graph_dict.keys():
        if v == 0:
            continue

        for i in range(v):
            if v in graph_dict[i].keys():
                incoming_dict[v].add(i)

    return incoming_dict

def generateLongestNetworkPath(graph_dict):
    """
    Find the longest network path in a directed acylic graph. 
    """
    S = findAllSources(graph_dict)
    #L = kahn_sort(graph_dict, S)
    L = getInterval(graph_dict)
    I = getIncomingDict(graph_dict)
    
    #initialise dictionary
    longest_dict = dict.fromkeys(graph_dict.keys())
    for key in graph_dict.keys():
        longest_dict[key] = {'dist': 0.0, 'bestParent': None}
    
    for v in L:
        #skip if v is a source
        if v in S:
            continue
        
        else:
            tempDist = {longest_dict[parent]['dist']+1: parent for parent in I[v]}
            temp = max(tempDist.keys())
            if temp > longest_dict[v]['dist']:
                longest_dict[v]['dist'] = temp
                longest_dict[v]['bestParent'] = tempDist[temp]

    return longest_dict

def getLongestPath(graph_dict, target):
    longest_dict = generateLongestNetworkPath(graph_dict)
    longestPath = []
    u = target
    if longest_dict[u]['bestParent'] != None or u == 0:
        while u:
            longestPath.append(u)
            u = longest_dict[u]['bestParent']

    longestPath.reverse()
    return longestPath, longest_dict[target]['dist']   