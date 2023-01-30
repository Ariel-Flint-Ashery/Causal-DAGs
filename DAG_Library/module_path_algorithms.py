# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 23:26:37 2022

@authors: Kevin & Ariel

This is a prototype module for Directed Acyclic Graph projects.
This module contains path searching algorithms.
"""
import matplotlib.pyplot as plt
import numpy as np
import copy 

def short_long_paths(graph_dict, edge_list = None, inv_graph_dict = None, target = None):
    """
        Shortest and Longest path search by labelling the distances of each node from the origin. 
        Shortest path obtained by tracking back from the shortest distance at the target, choosing
        edges to nodes with -1 distance from the current node.
        Longest path obtained by tracking back from the longest distance at the target, choosing
        edges to nodes with -1 distance from the current node.
        
        Input:
            graph_dict: Graph represented as an adjacency list in the form of a python dictionary
            e.g {0: [1, 2, 3],
                  1: [2, 3],
                  2: [4, 6]
                  ...}
            
            edge_list: Graph represented as an edge list, i.e [(0,1), (0,3), ...]
            
            inv_graph_dict: Inverse graph, i.e the graph with edges pointing backwards
            
            target: target node for the path to reach
        
        Output:
            Yields a generator listing all paths from the source to the target
        """
    if inv_graph_dict == None:
        inv_graph_dict = {u:{} for u in graph_dict}
        if edge_list == None:
            edge_list = [(u,v) for u in graph_dict for v in graph_dict[u]]
        for e in edge_list:    
            inv_graph_dict[e[1]][e[0]] = {}
        
    target = len(graph_dict) - 1
    queue = [{0}]
    node_dist = {u:set() for u in graph_dict}
    current_distance = 0 
    
    while queue:    # create a dictionary where key:values pairs are node:{distances from origin}
        next_generation = set()
        parents = queue[0]
        for parent in parents:
            children = graph_dict[parent]
            for child in children:
                next_generation.add(child)
            node_dist[parent].add(current_distance)
        current_distance += 1
        queue.pop(0)
        if len(next_generation) == 0:
            continue
        else:
            queue.append(next_generation)
            
    long_path_dist = max(node_dist[target])
    short_path_dist = min(node_dist[target])
    
    long_queue = [target]
    while long_queue:
        current = long_queue[-1]
        long_path_dist -= 1
        if long_path_dist == 0:
            long_queue.append(0)
            break
        parents = iter(inv_graph_dict[current])
        while parents:
            parent = next(parents, None)
            if long_path_dist in node_dist[parent]:
                parents = None
        long_queue.append(parent)
    
    short_queue = [target]
    while short_queue:
        current = short_queue[-1]
        short_path_dist -= 1
        if short_path_dist == 0:
            short_queue.append(0)
            break
        parents = iter(inv_graph_dict[current])
        while parents:
            parent = next(parents, None)
            if short_path_dist in node_dist[parent]:
                parents = None
        short_queue.append(parent)
    short_queue.reverse()
    long_queue.reverse()
        
    return short_queue, long_queue

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

def DFS_percolating(graph_dict, target = None, source = 0):
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
        
    if source not in graph_dict:
        return False
    
    visited = dict.fromkeys([source])
    queue = [c for c in graph_dict[source]]
    
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

def DFS(graph_dict, source=0):
    if source not in graph_dict:
        raise ValueError('Source not in graph')
    
    visited = dict.fromkeys(graph_dict)
    for key in visited.keys():
        visited[key] = False
    visited[source] = True  
    queue = [c for c in graph_dict[source]]
    while queue:
        child = queue[-1]
        queue.pop()
        if visited[child]:
            continue
        
        else:
            visited[child] = True
            children = [c for c in graph_dict[child]]
            for c in children:
                queue.append(c)
    
    return visited
                

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
    distance = 0.0
    
    for i in range(len(path)-1):
        distance += graph_dict[path[i]][path[i+1]].values() #[-1]

    return distance, len(path) #geometric distance, network distance

def pathJaggy(graph_dict, pos, path):
    theta = 0
    
    for i in range(len(path) - 2):
        u = pos[path[i+1]] - pos[path[i]]
        v = pos[path[i+2]] - pos[path[i+1]]
        grad_u = u[1]/u[0]
        grad_v = v[1]/v[0]
        sign = 1
        if grad_v < grad_u:
            sign = -1
        theta += sign * np.arccos(np.dot(u, v)/(np.linalg.norm(pos[-1]) * np.linalg.norm(u)))
    
    return theta/(len(path) - 1)

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
    sources = [0]
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
    
    if len(graph_dict)-1 in sources:
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

def getInverseGraph(graph_dict):
    inv_graph_dict = {u:{} for u in graph_dict}
    edge_list = [(u,v) for u in graph_dict for v in graph_dict[u]]
        #None #need to actually code a way to get edge list from graph dict
    for e in edge_list:    
        inv_graph_dict[e[1]][e[0]] = {}
    
    return inv_graph_dict
            
def getInterval(graph_dict):
    """
    Input:
        graph_dict

    Output:
        L: Topologically sorted (by Kahn's algorithm) interval between source and target.
    """
    #check if node connects to source:
    visited = DFS(graph_dict) #key:val = node:in interval?
    
    #perform topological sort
    S = [0] #we only consider one source node
    G = copy.deepcopy(graph_dict)
    I = getIncomingDict(graph_dict, visited)
    i = copy.deepcopy(I)
    L = []
    while S: #while S is not empty, will return True
        n = S.pop()
        L.append(n)
        #g = copy.deepcopy(graph_dict)
        for m in graph_dict[n]:
            G[n].pop(m)
            i[m].remove(n)
            if i[m] == set(): #if m has no more incoming edges
                S.append(m)

    return L, I

def getStrictInterval(graph_dict, s = 0, t = None):
    """
    Return the strict interval for a directed network between two nodes.
        Input:
            graph_dict: adjacency dictionary for the graph.
            s: source node.
            t: target node.

        Output:
            interval_dict: adjacency dictionary for the graph within the interval.
            incoming_dict: incoming adjacency dictionary for the graph within the interval.
    """
    if t == None:
        t = len(graph_dict) - 1

    visited = DFS(graph_dict, s) #key:val = node:in interval?
    inv_graph = getInverseGraph(graph_dict)
    incoming_dict = getIncomingDict(graph_dict)
    inv_visited = DFS(inv_graph, t)
    interval_dict = copy.deepcopy(graph_dict)
    incoming_copy = copy.deepcopy(incoming_dict)
    for v in graph_dict.keys():
        if visited[v] == False or inv_visited[v] == False:
            for c in graph_dict[v]:
                incoming_dict[c].pop(v, None)

            for p in incoming_copy[v]:
                interval_dict[p].pop(v, None)
            
            interval_dict.pop(v, None)
        
    return interval_dict, incoming_dict

def getIncomingDict(graph_dict, interval_dict = None): #we might want to move this to geometric graph module?
    """
    Return an adjacency dictionary for incoming edges.
        Input:
            graph_dict: adjacency dictionary of entire graph for edge between u and v.
            Here, keys indicate u and values indicate v. key:val = node:outgoing nodes

        Output:
            incoming_dict: adjacency dictionary of entire graph for edge between u and v.
            Here, keys indicate v and values indicate u. key:val = node:incoming nodes

    """
    incoming_dict = dict.fromkeys(graph_dict.keys())
    for key in graph_dict.keys():
        incoming_dict[key] = set()
    
    for v in graph_dict.keys():
        if v == 0:
            continue

        for i in range(v):
            if v in graph_dict[i].keys():
                if interval_dict == None:
                    incoming_dict[v].add(i)
                else:
                    if interval_dict[v] == True and interval_dict[i] == True:
                        incoming_dict[v].add(i)

    return incoming_dict

def generateLongestNetworkPath(graph_dict):
    """
    Find the longest network path in a directed acylic graph. 
    """
    S = findAllSources(graph_dict)
    print('sources found')
    #L = kahn_sort(graph_dict, S)
    L, I = getInterval(graph_dict)
    print('Interval complete')
    #print(L)
    # I = getIncomingDict(graph_dict)
    # print('Incoming complete')
    
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

def getLongestPath(graph_dict):
    target = len(graph_dict) - 1
    longest_dict = generateLongestNetworkPath(graph_dict)
    longestPath = []
    u = target
    if longest_dict[u]['bestParent'] != None or u == 0:
        while u:
            longestPath.append(u)
            u = longest_dict[u]['bestParent']
            print(u)
    longestPath.append(0)
    longestPath.reverse()
    return longestPath, longest_dict[target]['dist']   

#def getGraphMeasures(graph_dict):
"""
- longest path
- shortest path
- greedy path
- for each path: geometric & network distance
- jaggedness: wrt line and path
- Largest separation from straight line
- average/total separation from straight line: Euclidean + Lp
"""

#def plotter(X, p_vals, iterations = 1):
"""
bar plots of each metric, for each p value. 
"""
    