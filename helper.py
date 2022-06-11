from xmlrpc.client import Boolean
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

def create_tsp_graph(num_nodes : int) -> np.array:
    """Create random TSP graph

    Args:
        num_nodes (int): Number of cities in the TSP problem 

    Returns:
        np.array: weight matrix of the TSP 
    """
    #create random matrix and assure that there are no 0 costs
    graph = np.random.rand(num_nodes,num_nodes)*100 + 0.1 

    #set diagonal values to 0 
    for i in range(num_nodes):
        graph[i,i] = 0.

    # assure that matrix is symmetric (tsp is undirected)
    graph = (graph + graph.T) /2
    return graph


def cost(graph : np.array , path :List) -> float:
    """ takes the graph matrix and a path and returns its cost

    Args:
        graph (np.array): TSP graph, represented as symmetric np.array
        path (List): Order in which cities are visited in integer format, e.g. [0, 1, 3, 2]

    Returns:
        float: cost of that path
    """
    cost = 0.

    # start with first and second city and traverse the whole path 
    last_city = path[0]
    for next_city in path[1:]:
        cost += graph[last_city,next_city]
        last_city = next_city

    # add path from last to first city
    cost += graph[last_city,path[0]]
    return cost

def plot_tsp_graph(tsp_matrix : np.array) -> None:
    """Plot the weighted graph of a TSP

    Args:
        tsp_matrix (np.array): TSP graph, represented as symmetric np.array
    """
    G = nx.Graph()
    for i in range(tsp_matrix.shape[0]):
        for j in range(i+1,tsp_matrix.shape[0]):
            G.add_edge(f"{i}", f"{j}", weight=round(tsp_matrix[i,j],2))

    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, width=3,)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def bitstring_to_path(bitstring:str,return_as_string=False)->List:
    """Converts bitstring encoding to path, e.g. for 3 cities: 010100001 -> [1,0,2] 

    Args:
        bitstring (str): bitstring of a path, size ncities**2
        return_as_string (Boolean): Whether the path should be returned as string, for plotting purposes
    Returns:
        List: Order in which cities are visited in integer format, e.g. [0, 1, 3, 2], None if bitstring does not represent a valid path
    """
    # maybe include checks whether this is actually an integer ? 
    ncities = int(len(bitstring)**0.5)

    path = []
    
    try:
        for i in range(0,len(bitstring),ncities):

            if bitstring[i:i+ncities].count('1') == 1:
              path.append(bitstring[i:i+ncities].index('1'))
            
            # invalid path, eg. 011000111
            else:
                return None
            
    # not a valid path, e.g. 0000000
    except ValueError:
        return None

    if return_as_string:
        path = str(path)
    return path 

