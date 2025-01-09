from graph import Graph
from qubo_problem import QUBOProblem
from plot_solutions import draw_solution, sample_and_plot_histogram, interpret_solution
from config import Config
import numpy as np


def main():
    # Initial configurations
    config = Config()

    # Example usage:
    N = 4
    weight_range = (10, 100)
    seed = 78
    max_degree = 2

    # Instantiate and create the graph
    graph = Graph(N, weight_range, seed)
    graph.draw()

    # Generate the adjacency matrix
    adj_matrix = np.zeros((N, N), dtype=int)
    for (u, v, data) in graph.G.edges(data=True):
        adj_matrix[u][v] = data['weight']
        adj_matrix[v][u] = data['weight']  # Ensure symmetry    

    # Find the mst using kruskal
    mst_edges, total_weight = graph.find_mst_kruskal()
    graph.plot_mst(mst_edges, title="Kruskal MST")

    # find the dcmst by brute force
    dcmst_edges, dcmst_cost = graph.find_dcmst_brute_force(max_degree)
    if dcmst_edges is not None:
        graph.plot_mst(dcmst_edges, mst_color='red', title=f"DC-MST (max_degree={max_degree})")

    # Configure and solve the QUBO problem
    qubo_problem = QUBOProblem(graph.G, config)
    qubo_problem.configure_variables()
    qubo_problem.define_objective_function()
    qubo_problem.add_constraints()
    samples = qubo_problem.solve_problem()

    solution = qubo_problem.solution

    # Visualize the solution

    valid_solutions = sample_and_plot_histogram(
        samples,
        adj_matrix=adj_matrix,
        N=N,
        Delta=max_degree,
        interpret_solution_fn=interpret_solution,
        top_n=30
    )    

    # Draw the most sampled
    draw_solution(graph, solution)
    
    import networkx as nx
    import matplotlib.pyplot as plt


if __name__ == "__main__":
    main()