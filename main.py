from graph import Graph
from qubo_problem import DCMST_QUBO
from plot_solutions import draw_solution, sample_and_plot_histogram, interpret_solution
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
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

    # Create degree_constraints dictionary with the same max degree for all nodes
    degree_constraints = {node: max_degree for node in graph.G.nodes()}

    # Configure and solve the QUBO problem
    qubo_problem = DCMST_QUBO(graph.G, degree_constraints, config)
    qubo_problem.configure_variables()
    qubo_problem.define_objective_function()
    qubo_problem.add_constraints()

    # Print the number of qubits necessary to solve the problem
    qubo_problem.print_number_of_qubits()

    optimizer = COBYLA()
    p = 1  # QAOA circuit depth

    samples = qubo_problem.solve_problem(optimizer, p)

    # This is just the test if the correct answers are the same as the most sampled ones. 
    # def format_qaoa_samples(samples, max_len: int = 10):
    #     qaoa_res = []
    #     for s in samples:
    #         bitstring = ''.join(str(int(v)) for v in s.x)
    #         # bitstring = bitstring[::-1]            
    #         qaoa_res.append((bitstring, s.fval, s.probability))

    #     res = sorted(qaoa_res, key=lambda x: -x[2])[0:max_len]

    #     return [(_[0] + f": value: {_[1]:.3f}, probability: {1e2*_[2]:.1f}%") for _ in res]


    # print(format_qaoa_samples(samples))   

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


if __name__ == "__main__":
    main()