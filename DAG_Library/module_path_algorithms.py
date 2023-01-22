# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 23:26:37 2022

@authors: Kevin & Ariel

This is a prototype module for Directed Acyclic Graph projects.
This module contains path searching algorithms.
"""
import matplotlib.pyplot as plt
import numpy as np

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
            None #need to actually code a way to get edge list from graph dict
        for e in edge_list:    
            inv_graph_dict[e[1]][e[0]] = {}
        
    target = len(graph_dict) - 1
    print(inv_graph_dict[target])
    queue = [0]
    visited = {}
    node_dist = {u:{} for u in graph_dict}
    node_dist[0][0] = None
    
    while queue:    # create a dictionary where key:values pairs are node:{distances from origin}
        current = queue[0]
        children = graph_dict[current]
        for child in children:
            dist = [d + 1 for d in node_dist[current].keys()]
            for d in dist:
                node_dist[child][d] = None
            if child in visited:
                continue
            else:
                visited[child] = None
                queue.append(child)
        queue.pop(0)
    
    long_path_dist = max(node_dist[target])
    short_path_dist = min(node_dist[target])
    
    long_queue = [target]
    while long_queue:   # choose nodes starting from the target and working backwards
        current = long_queue[-1]
        long_path_dist -= 1
        if long_path_dist == 0:
            long_queue.append(0)
            break
        parents = iter(inv_graph_dict[current])
        while parents:
            parent = next(parents, None)
            print(long_path_dist)
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