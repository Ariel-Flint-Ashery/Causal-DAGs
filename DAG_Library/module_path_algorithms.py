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