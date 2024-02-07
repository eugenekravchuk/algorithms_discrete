import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
import numpy as np
import heapq
from collections import defaultdict
import time
from tqdm import tqdm
from networkx.algorithms import floyd_warshall_predecessor_and_distance

def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               directed: bool = False,
                               draw: bool = False):
    """
    Generates a random graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted (in case of undirected graphs)
    """

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    edges = combinations(range(num_of_nodes), 2)
    G.add_nodes_from(range(num_of_nodes))
    
    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        if random.random() < 0.5:
            random_edge = random_edge[::-1]
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)
                
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(-5, 20)
                
    if draw: 
        plt.figure(figsize=(10,6))
        if directed:
            # draw with edge weights
            pos = nx.spring_layout(G)
            nx.draw(G,pos, node_color='lightblue', 
                    with_labels=True,
                    node_size=500, 
                    arrowsize=20, 
                    arrows=True)
            labels = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
            
        else:
            pos = nx.spring_layout(G)
            nx.draw(G,pos, node_color='lightblue', 
                    with_labels=True,
                    node_size=500)
            labels = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
        
    return G

def convert_to_matrix_for_prim(graph):
    directed = isinstance(graph, nx.DiGraph)
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    matrix = [[0 for i in range(len(graph.nodes))] for j in range(len(graph.nodes))]

    if not directed:
        edge_weights_copy = dict()
        for edge in edge_weights:
            edge_weights_copy[edge] = edge_weights[edge]
            edge_weights_copy[edge[::-1]] = edge_weights[edge]

        edge_weights = edge_weights_copy

    for edge in edge_weights:
        matrix[edge[0]][edge[1]] = edge_weights[edge]

    return matrix

def convert_to_matrix_for_warshall(graph, inf):
    directed = isinstance(graph, nx.DiGraph)
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    matrix = [[inf for i in range(len(graph.nodes))] for j in range(len(graph.nodes))]

    if not directed:
        edge_weights_copy = dict()
        for edge in edge_weights:
            edge_weights_copy[edge] = edge_weights[edge]
            edge_weights_copy[edge[::-1]] = edge_weights[edge]

        edge_weights = edge_weights_copy

    for edge in edge_weights:
        matrix[edge[0]][edge[1]] = edge_weights[edge]
    
    for i in range(len(matrix)):
        matrix[i][i] = 0

    return matrix

def prims_algorithm(graph):

    directed = isinstance(graph, nx.DiGraph)

    N = len(graph)
    inf = float('inf')
    selected_node = [0]*N
    no_edge = 0
    selected_node[0] = True
    matrix = convert_to_matrix_for_prim(graph)
    weighted_list = []
    while(no_edge < N - 1):
        minimum = inf
        x = 0
        y = 0
        for i in range(N):
            if selected_node[i]:
                for j in range(N):
                    if ((not selected_node[j]) and matrix[i][j]):
                        if minimum > matrix[i][j]:
                            minimum = matrix[i][j]
                            x = i
                            y = j
        weighted_list.append((x, y, matrix[x][y]))
        selected_node[y] = True
        no_edge += 1


    if directed:
        new_graph = nx.DiGraph()
        new_graph.add_weighted_edges_from(weighted_list)
    else:
        new_graph = nx.Graph()
        new_graph.add_weighted_edges_from(weighted_list)

    return new_graph

def floyd_warshall_algo(graph):

    inf = 9999

    matrix = convert_to_matrix_for_warshall(graph, inf)

    n = len(matrix)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])

    for row in matrix:
        for elem in row:
            if elem > inf:
                return "Negative cycle detected"
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] > inf - 1000:
                matrix[i][j] = "inf"
    return matrix

def count_time(function):
    NUM_OF_ITERATIONS = 1000
    time_taken = 0
    for _ in tqdm(range(NUM_OF_ITERATIONS)):
        
        # note that we should not measure time of graph creation
        G = gnp_random_connected_graph(100, 0.4, False)
        
        start = time.time()
        function(G)
        end = time.time()
        
        time_taken += end - start

    return time_taken / NUM_OF_ITERATIONS

G = gnp_random_connected_graph(5, 0.5, True, True)

count_time(floyd_warshall_algo)
