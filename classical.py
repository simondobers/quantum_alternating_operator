from typing import List, Tuple
import numpy as np
from itertools import permutations
import sys

def create_tsp_graph(num_nodes : int) -> np.array:
    """Create random TSP graph

    Args:
        num_nodes (int): Number of cities in the TSP problem 

    Returns:
        np.array: weight matrix of the TSP 
    """
    #create random matrix and assure that there are no 0 costs
    graph = np.random.rand(num_nodes,num_nodes) + 0.1 

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
    cost += graph[last_city,0]
    return cost


def solve_tsp_classical(graph : np.array , starting_city=0) -> Tuple :
    """Check all possible combinations and return the one with the lowest cost 

    Args:
        graph (np.array): TSP graph, represented as symmetric np.array
        starting_city (int): City from which the salesperson should start 

    Returns:
        (Path,Cost): (Order in which cities are visited in integer format, e.g. [0, 1, 3, 2] :List) , Cost of that path)
    """

    best_cost = np.inf
    best_path = []

    #get all other cities except for the starting one
    remaining_cities = [i for i in range(graph.shape[0]) if i != starting_city]

    # check all possible permutations
    for permutation in permutations(remaining_cities):
        current_path = [starting_city] + list(permutation)
        current_cost = cost(graph,current_path)

        # check if current path is better than currently best path 
        if current_cost < best_cost :
            best_cost = current_cost
            best_path = current_path

    return best_path,best_cost

    

if __name__ == "__main__":
    """ Generate random TSP and solve for best path

    Args : 
        num_cities: Number of cities in the TSP 
    """
    _,num_cities = sys.argv
    num_cities = int(num_cities)

    tsp_grah = create_tsp_graph(num_cities)

    print("Weight matrix")
    print(tsp_grah)
    print("-------------")

    path,cost = solve_tsp_classical(tsp_grah)
    print(f"Best path:{path} with cost={cost}")